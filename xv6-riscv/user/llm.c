#include "kernel/types.h"
#include "user/user.h"
#include "user/xv6_string.h"
#include "user/xv6_maths.h"
#include "user/udp_client.h"
#include "user/sha256.h"
#include "user/xv6_stdlib.h"


// Replace standard math functions with xV6_maths implementations
#define sqrtf xv6m_sqrtf
#define expf xv6m_expf
#define powf xv6m_powf
#define cosf xv6m_cosf
#define sinf xv6m_sinf
#define fabs xv6m_fabsf

// string fns
#define  memcpy xv6_memcpy
#define  memset xv6_memset
#define  strcpy xv6_strcpy
#define  strcmp xv6_strcmp
#define strcat xv6_strcat

// Common macros
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

// File descriptors in xv6
#define stdout 1
#define stderr 2

const char* g_test_name = "RUN";

// Timing variables for benchmarking
unsigned long long matmul_cycles = 0;
unsigned long long activation_cycles = 0;
unsigned long long attention_cycles = 0;
unsigned long long ffn_cycles = 0;
unsigned long long sampling_cycles = 0;

// printf macros

#define printf(stream, fmt, ...) do { \
    if (stream == stderr) printf("Error: " fmt "\n" , ##__VA_ARGS__); \
    else printf(fmt, ##__VA_ARGS__); \
} while(0)


// Model configuration
typedef struct {
    int dim;           // transformer dimension
    int hidden_dim;    // for ffn layers
    int n_layers;      // number of layers
    int n_heads;       // number of query heads
    int n_kv_heads;    // number of key/value heads
    int vocab_size;    // vocabulary size
    int seq_len;       // max sequence length
} Config;

// Transformer weights
typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight;           // (layer, dim) rmsnorm weights
    float* rms_ffn_weight;           // (layer, dim)
    // weights for matmuls
    float* wq;                       // (layer, dim, dim)
    float* wk;                       // (layer, dim, dim)
    float* wv;                       // (layer, dim, dim)
    float* wo;                       // (layer, dim, dim)
    // weights for ffn
    float* w1;                       // (layer, hidden_dim, dim)
    float* w2;                       // (layer, dim, hidden_dim)
    float* w3;                       // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight;         // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real;            // (seq_len, dim/2)
    float* freq_cis_imag;            // (seq_len, dim/2)
    float* wcls;
} TransformerWeights;

// Run state (intermediate activations)
typedef struct {
    float *x;          // input to current layer, (seq_len, dim)
    float *xb;         // same, but inside a residual branch (seq_len, dim)
    float *xb2;        // an additional buffer (seq_len, dim)
    float *hb;         // hidden state for ffn (seq_len, hidden_dim)
    float *hb2;        // hidden state for ffn (seq_len, hidden_dim)
    float *q;          // query (seq_len, dim)
    float *k;          // key (seq_len, dim)
    float *v;          // value (seq_len, dim)
    float *att;        // attention scores (seq_len, seq_len)
    float *logits;     // output logits (vocab_size)
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

// Transformer model
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    // memory buffers that point to the allocated memory
    float* memory;
    int memory_size;
} Transformer;

// CPU frequency for converting cycles -> seconds
// TODO: set this to whatever your xv6 CPU freq is (e.g., 1e9 for 1 GHz)
// default: 10 MHZ
#define CPU_FREQ_HZ 10000000.0

// Program start timestamp (for TTFT + End-to-End)
unsigned long long g_program_start_cycles = 0;

// Reset hotspot cycle counters before each benchmark run
void reset_benchmark_counters(void) {
    matmul_cycles      = 0;
    activation_cycles  = 0;
    attention_cycles   = 0;
    ffn_cycles         = 0;
    sampling_cycles    = 0;
}

// ----------------------------------------------------------------------------
// Neural network operations

void rmsnorm(float* o, float* x, float* weight, int size) {
    unsigned long long start = gettime();
    
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f; // epsilon to avoid div by zero
    ss = 1.0f / sqrtf(ss);
    
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    
    activation_cycles += (gettime() - start);
}

void softmax(float* x, int size) {
    unsigned long long start = gettime();
    
    // find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
    
    activation_cycles += (gettime() - start);
}

void matmul(float* output, float* input, float* weight, int n, int d) {
    unsigned long long start = gettime();
   
    // W (d,n) @ x (n,) -> xout (d,)   
    
    //parallelized here
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += weight[i * n + j] * input[j];
        }
        output[i] = val;
    }
    
    matmul_cycles += (gettime() - start);
}

// ----------------------------------------------------------------------------
// Transformer operations

void transformer_forward(int token, int pos, Config* p, TransformerWeights* w, RunState* s) {
    // a few convenience variables
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    
    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(s->x, content_row, dim * sizeof(float));
    
    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = &(w->freq_cis_real[pos * head_size / 2]);
    float* freq_cis_imag_row = &(w->freq_cis_imag[pos * head_size / 2]);
    
    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // attention rmsnorm (counted in activation_cycles)
        rmsnorm(s->xb, s->x, &(w->rms_att_weight[l * dim]), dim);

        // === Attention timing: QKV + scores + output ===
        unsigned long long att_start = gettime();
        
        // qkv matmuls for this position
        matmul(s->q, s->xb, &(w->wq[l * dim * dim]), dim, dim);
        matmul(s->k, s->xb, &(w->wk[l * dim * dim]), dim, dim);
        matmul(s->v, s->xb, &(w->wv[l * dim * dim]), dim, dim);
        
        // apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++) {
            // get the q and k vectors for this head
            float* q = s->q + h * head_size;
            float* k = s->k + h * head_size;
            
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i += 2) {
                float q0 = q[i];
                float q1 = q[i + 1];
                float k0 = k[i];
                float k1 = k[i + 1];
                float fcr = freq_cis_real_row[i / 2];
                float fci = freq_cis_imag_row[i / 2];
                q[i]     = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i]     = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }
        
        // save key, value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset
        float* key_cache_row   = &(s->key_cache[loff + pos * dim]);
        float* value_cache_row = &(s->value_cache[loff + pos * dim]);
        memcpy(key_cache_row,   s->k, dim * sizeof(float));
        memcpy(value_cache_row, s->v, dim * sizeof(float));
        
        // multi-head attention
        // parallelized here
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = &(s->key_cache[loff + t * dim + h * head_size]);
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }
            
            // softmax the scores to get attention weights
            softmax(att, pos + 1);
            
            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = &(s->value_cache[loff + t * dim + h * head_size]);
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate to xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        
        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, &(w->wo[l * dim * dim]), dim, dim);

        // record attention time (QKV + scores + output)
        attention_cycles += (gettime() - att_start);
        
        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        
        // ffn rmsnorm (activations)
        rmsnorm(s->xb, s->x, &(w->rms_ffn_weight[l * dim]), dim);
        
        unsigned long long ffn_start = gettime();
        
        // FFN: w1, w3
        matmul(s->hb,  s->xb, &(w->w1[l * dim * hidden_dim]), dim, hidden_dim);
        matmul(s->hb2, s->xb, &(w->w3[l * dim * hidden_dim]), dim, hidden_dim);
        
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        
        // FFN: w2
        matmul(s->xb, s->hb, &(w->w2[l * hidden_dim * dim]), hidden_dim, dim);
        
        ffn_cycles += (gettime() - ffn_start);
        
        // residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
    
    // final rmsnorm
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    
    // classifier into logits
    matmul(s->logits, s->x, w->token_embedding_table, dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// Sampling utilities

int sample(float* probabilities, int n, float coin) {
    unsigned long long start = gettime();
    
    // sample index from probabilities, they must sum to 1
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            sampling_cycles += (gettime() - start);
            return i;
        }
    }
    
    sampling_cycles += (gettime() - start);
    return n - 1; // in case of rounding errors
}

// Comparator for sorting floats in descending order
int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

// Apply top-p (nucleus) sampling
// probabilities: array of logits or probabilities (normalized)
// n: size of array
// top_p: threshold (0 < top_p <= 1)
int apply_top_p(float* probabilities, int n, float top_p) {
    // Make a copy for sorting
    float* sorted = (float*)malloc(n * sizeof(float));
    if (!sorted) return -1; // malloc failed
    memcpy(sorted, probabilities, n * sizeof(float));

    // Sort descending
    xv6_qsort(sorted, n, sizeof(float), cmp_desc);

    // Determine cutoff index for cumulative probability
    float cum_prob = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cum_prob += sorted[i];
        cutoff = i;
        if (cum_prob >= top_p) break;
    }

    // Mask probabilities below cutoff
    float threshold = sorted[cutoff];
    for (int i = 0; i < n; i++) {
        if (probabilities[i] < threshold) probabilities[i] = 0.0f;
    }

    // Normalize remaining probabilities
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probabilities[i];
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) probabilities[i] /= sum;
    }

    free(sorted);
    return 0;
}


int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

Tokenizer* build_tokenizer(int vocab_size) {
    int tokenizer_size;
    char* tokenizer_data = fetch_tokenizer(&tokenizer_size);
    if (tokenizer_data == 0) {
        printf(stderr,"Failed to load tokenizer\n");
        exit(1);
    }
    
    Tokenizer* t = (Tokenizer*)malloc(sizeof(Tokenizer));
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    
    // Initialize byte_pieces
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    
    // Parse: [max_token_length][score][len][string]...
    char* ptr = tokenizer_data;
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);
    
    for (int i = 0; i < vocab_size; i++) {
        // Read score
        memcpy(&t->vocab_scores[i], ptr, sizeof(float));
        ptr += sizeof(float);
        
        // Read length
        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);
        
        // Read string
        t->vocab[i] = (char*)malloc(len + 1);
        memcpy(t->vocab[i], ptr, len);
        t->vocab[i][len] = '\0';
        ptr += len;
    }
    
    free(tokenizer_data);
    return t;
}
void free_tokenizer(Tokenizer *tokenizer, int vocab_size) {
    if (!tokenizer) return;

    // Free each vocab string
    if (tokenizer->vocab) {
        for (int i = 0; i < vocab_size; i++) {
            if (tokenizer->vocab[i]) {
                free(tokenizer->vocab[i]);
            }
        }
        free(tokenizer->vocab);
    }

    // Free vocab_scores array
    if (tokenizer->vocab_scores) {
        free(tokenizer->vocab_scores);
    }

    // Finally free the tokenizer struct itself
    free(tokenizer);
}


// ----------------------------------------------------------------------------
// Model loading

// ----------------------------------------------------------------------------
// Initialize weight pointers from contiguous memory block
void init_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    
    w->rms_final_weight = ptr;
    ptr += p->dim;
    
    // Skip freq_cis
    ptr += p->seq_len * head_size / 2;  // skip freq_cis_real
    ptr += p->seq_len * head_size / 2;  // skip freq_cis_imag
    
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}
// ----------------------------------------------------------------------------
// Allocate contiguous memory for all weights and initialize pointers
// ----------------------------------------------------------------------------
void read_checkpoint(Config* config, TransformerWeights* weights, 
                     float** memory, int* memory_size) {
                     
    // ----------------------
    // Fetch raw model data from server
    // ----------------------
    int size_bytes = 0;
    char* raw_weights = fetch_model_weights(&size_bytes);
    if (!raw_weights) {
        printf(stderr,"Failed to fetch model weights from server\n");
        exit(1);
    }
    
    // ----------------------
    // Extract Config from first bytes of file
    // ----------------------
    if (size_bytes < sizeof(Config)) {
        printf(stderr,"Model file too small to contain Config struct\n");
        free(raw_weights);
        exit(1);
    }
    
    memcpy(config, raw_weights, sizeof(Config));
    
    // Handle shared_weights flag (negative vocab_size means separate wcls)
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = fabs(config->vocab_size);  // Make positive
    
    int dim        = config->dim;
    int hidden_dim = config->hidden_dim;
    int n_layers   = config->n_layers;
    int n_heads    = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int vocab_size = config->vocab_size;
    int seq_len    = config->seq_len;
    int head_size  = dim / n_heads;
    
    printf(stdout,"Loaded model config:\n");
    printf(stdout,"  dim=%d, hidden_dim=%d, n_layers=%d\n", dim, hidden_dim, n_layers);
    printf(stdout,"  n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n", 
           n_heads, n_kv_heads, vocab_size, seq_len);
    printf(stdout,"  shared_weights=%d\n", shared_weights);
    
    // ----------------------
    // Calculate total memory required for weights only
    // Use unsigned long long to avoid overflow for large models
    // ----------------------
    unsigned long long total_floats = 0;
    
    // Token embeddings
    total_floats += (unsigned long long)vocab_size * dim;
    
    // Attention weights
    total_floats += (unsigned long long)n_layers * dim;  // rms_att_weight
    
    // QKV projection weights
    total_floats += (unsigned long long)n_layers * dim * (n_heads * head_size);     // wq
    total_floats += (unsigned long long)n_layers * dim * (n_kv_heads * head_size);  // wk
    total_floats += (unsigned long long)n_layers * dim * (n_kv_heads * head_size);  // wv
    
    // Output projection
    total_floats += (unsigned long long)n_layers * (n_heads * head_size) * dim;     // wo
    
    // FFN weights
    total_floats += (unsigned long long)n_layers * dim;                // rms_ffn_weight
    total_floats += (unsigned long long)n_layers * dim * hidden_dim;   // w1
    total_floats += (unsigned long long)n_layers * hidden_dim * dim;   // w2
    total_floats += (unsigned long long)n_layers * dim * hidden_dim;   // w3
    
    // Final layer norm
    total_floats += (unsigned long long)dim;                           // rms_final_weight
    
    // RoPE frequencies (these are skipped but still in file)
    total_floats += (unsigned long long)seq_len * head_size / 2;      // freq_cis_real
    total_floats += (unsigned long long)seq_len * head_size / 2;      // freq_cis_imag
    
    // Classifier weights (only if not shared)
    if (!shared_weights) {
        total_floats += (unsigned long long)vocab_size * dim;          // wcls
    }
    
    *memory_size = (int)total_floats;
    
    printf(stdout,"Calculated weights size: %d floats (%llu bytes)\n", 
           *memory_size, (unsigned long long)*memory_size * sizeof(float));
    
    // ----------------------
    // Check that file size matches expected size
    // ----------------------
    unsigned long long expected_bytes = sizeof(Config) + total_floats * sizeof(float);
    
    printf(stdout,"Expected file size: %llu bytes\n", expected_bytes);
    printf(stdout,"Actual file size: %d bytes\n", size_bytes);
    
    if (fabs(size_bytes - expected_bytes) > 1024) {  // Allow 1KB tolerance
        printf(stderr,"File Size mismatch! Expected %llu, got %d\n", 
               expected_bytes, size_bytes);
        free(raw_weights);
        exit(1);
    }
    
    // ----------------------
    // Allocate contiguous memory for weights
    // ----------------------
    *memory = (float*)malloc(*memory_size * sizeof(float));
    if (!*memory) {
        printf(stderr,"malloc failed for model weights (%d floats)\n", *memory_size);
        free(raw_weights);
        exit(1);
    }
    
    // ----------------------
    // Copy weights from raw buffer (after Config) into float memory
    // ----------------------
    memcpy(*memory, raw_weights + sizeof(Config), *memory_size * sizeof(float));
    free(raw_weights);
    
    // ----------------------
    // Initialize pointers inside TransformerWeights
    // ----------------------
    init_weights(weights, config, *memory, shared_weights);
    
    head_size = config->dim / config->n_heads;
    weights->freq_cis_real = (float*)malloc(config->seq_len * head_size / 2 * sizeof(float));
    weights->freq_cis_imag = (float*)malloc(config->seq_len * head_size / 2 * sizeof(float));
    
    for (int pos = 0; pos < config->seq_len; pos++) {
        for (int i = 0; i < head_size / 2; i++) {
            float freq = 1.0f / xv6m_powf(10000.0f, (2.0f * i) / (float)head_size);
            float val = pos * freq;
            weights->freq_cis_real[pos * (head_size / 2) + i] = xv6m_cosf(val);
            weights->freq_cis_imag[pos * (head_size / 2) + i] = xv6m_sinf(val);
        }
    }
    
    printf(stdout,"Model weights loaded successfully into memory\n");
}
// ----------------------------------------------------------------------------
// Allocate run-state buffers (seq_len aware)
void malloc_run_state(RunState* s, Config* p) {
    int seq_len = p->seq_len;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_heads = p->n_heads;
    int n_layers = p->n_layers;

    s->x = (float*)malloc(seq_len * dim * sizeof(float));
    s->xb = (float*)malloc(seq_len * dim * sizeof(float));
    s->xb2 = (float*)malloc(seq_len * dim * sizeof(float));
    s->hb = (float*)malloc(seq_len * hidden_dim * sizeof(float));
    s->hb2 = (float*)malloc(seq_len * hidden_dim * sizeof(float));
    s->q = (float*)malloc(seq_len * dim * sizeof(float));
    s->k = (float*)malloc(seq_len * dim * sizeof(float));
    s->v = (float*)malloc(seq_len * dim * sizeof(float));
    s->att = (float*)malloc(seq_len * n_heads * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));
    s->key_cache = (float*)malloc(n_layers * seq_len * dim * sizeof(float));
    s->value_cache = (float*)malloc(n_layers * seq_len * dim * sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || 
        !s->v || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        printf(stderr,"malloc failed!\n");
        exit(1);
    }
}


void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}



void free_transformer(Transformer* t) {
    free(t->memory);
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Main generation function

void generate(Transformer* model, Tokenizer* t, char* prompt, 
              int steps, float temperature, float topp, unsigned long long seed) {
    
    Config* p = &model->config;
    RunState* s = &model->state;
    
    // **TOKENIZATION**
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    
    // BOS token
    prompt_tokens[num_prompt_tokens++] = 1;
    
    // Tokenize each character (simplified)
    for (int i = 0; prompt[i] != '\0'; i++) {
        // Map character to token (3+ because 0=unk, 1=bos, 2=eos)
        prompt_tokens[num_prompt_tokens++] = (unsigned char)prompt[i] + 3;
    }
    
    int token = prompt_tokens[0];
    int pos = 0;

    // benchmarking counters
    int generated_tokens = 0;
    int ttft_measured = 0;
    unsigned long long ttft_cycles = 0;
    unsigned long long gen_start_cycles = 0;

    // Inference start: right before first forward pass
    unsigned long long infer_start_cycles = gettime();


    printf(stdout,"<start>\n");

    // Now: steps = max number of *generated* tokens (not total positions)
    while (pos < p->seq_len && generated_tokens < steps) {
        // Forward pass
        transformer_forward(token, pos, p, &model->weights, s);
        
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            // Sample
            if (temperature == 0.0f) {
                next = argmax(s->logits, p->vocab_size);
            } else {

                 // Apply softmax
                float max_val = s->logits[0];
                for (int i = 1; i < p->vocab_size; i++) {
                    if (s->logits[i] > max_val) max_val = s->logits[i];
                }
                float sum = 0.0f;
                for (int i = 0; i < p->vocab_size; i++) {
                    s->logits[i] = xv6m_expf((s->logits[i] - max_val) / temperature);
                    sum += s->logits[i];
                }
                for (int i = 0; i < p->vocab_size; i++) {
                    s->logits[i] /= sum;
                }
                apply_top_p(s->logits, p->vocab_size, topp);
                
                // Sample
                float coin = (float)((seed >> 16) % 10000) / 10000.0f;
                seed = seed * 1664525ULL + 1013904223ULL;
                next = sample(s->logits, p->vocab_size, coin);
            }

            // First *generated* token -> TTFT
            if (!ttft_measured) {
                unsigned long long first_token_cycle = gettime();
                ttft_cycles = first_token_cycle - g_program_start_cycles;
                gen_start_cycles = first_token_cycle;
                ttft_measured = 1;
            }

            generated_tokens++;
        }
        
        // Decode token
        if (next < 3) {
            // Special tokens
            if (next == 2) break; // EOS
        } else if (next < 259) {
            // Byte tokens
            printf(stdout,"%c", (char)(next - 3));
        } else if (next < t->vocab_size && t->vocab[next]) {
            printf(stdout,"%s", t->vocab[next]);
        }
        
        token = next;
        pos++;
    }
    
    printf(stdout,"\n<end>\n");

    // End-to-end from program start
    unsigned long long end_cycles = gettime();
    unsigned long long e2e_cycles = end_cycles - g_program_start_cycles;

    // Inference-only time (prefill + decode loop)
    unsigned long long infer_end_cycles = end_cycles;
    unsigned long long infer_cycles_ull = infer_end_cycles - infer_start_cycles;

    if (gen_start_cycles == 0) {
        // In case we never generated any new tokens (edge case)
        gen_start_cycles = end_cycles;
    }
    unsigned long long gen_cycles = end_cycles - gen_start_cycles;

    double ttft_seconds = (double)ttft_cycles / CPU_FREQ_HZ;
    double e2e_seconds  = (double)e2e_cycles / CPU_FREQ_HZ;

    // TPS: (output_tokens - 1) / generation_time
    double tps = 0.0;
    if (generated_tokens > 1 && gen_cycles > 0) {
        double gen_seconds = (double)gen_cycles / CPU_FREQ_HZ;
        tps = (double)(generated_tokens - 1) / gen_seconds;
    }

    // Hotspot percentages (relative to inference time)
    double infer_cycles = (double)infer_cycles_ull;
    if (infer_cycles <= 0.0) infer_cycles = 1.0; // avoid div0

    double matmul_pct      = 100.0 * (double)matmul_cycles     / infer_cycles;
    double activation_pct  = 100.0 * (double)activation_cycles / infer_cycles;
    double attention_pct   = 100.0 * (double)attention_cycles  / infer_cycles;
    double ffn_pct         = 100.0 * (double)ffn_cycles        / infer_cycles;
    double sampling_pct    = 100.0 * (double)sampling_cycles   / infer_cycles;


    // === Structured benchmark output in the required format ===
    // NOTE: we'll pass test_name + seed from main (next section)
    extern const char* g_test_name;  // declared in main

    printf(stdout,"\n=== TEST RUN %s ===\n", g_test_name);
    printf(stdout,"Prompt:\n\"%s\"\n", prompt);
    printf(stdout,"Prompt Tokens: %d\n", num_prompt_tokens);
    printf(stdout,"Output Tokens: %d\n", generated_tokens);
    printf(stdout,"Temperature: %f\n", temperature);
    printf(stdout,"Seed: %llu\n", seed);

    printf(stdout,"PRIMARY METRICS:\n");
    printf(stdout,"TTFT: %llu cycles (%f seconds)\n",
           ttft_cycles, ttft_seconds);
    printf(stdout,"TPS: %f tokens/sec\n", tps);
    printf(stdout,"End-to-End: %llu cycles (%f seconds)\n",
           e2e_cycles, e2e_seconds);

    printf(stdout,"HOTSPOT BREAKDOWN (%% of inference time):\n");
    printf(stdout,"matmul(): %f\n", matmul_pct);
    printf(stdout,"Activations (expf, sqrtf): %f%%\n", activation_pct);
    printf(stdout,"Attention: %f%%\n", attention_pct);
    printf(stdout,"FFN: %f%%\n", ffn_pct);
    printf(stdout,"Sampling: %f%%\n", sampling_pct);

    printf(stdout,"NOTES: Overlapping categories; matmul is used inside Attention and FFN.\n");
    printf(stdout,"===================\n");

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// Main function

int main(int argc, char *argv[]) {

    // Program start time for TTFT & End-to-End
    g_program_start_cycles = gettime();

    // Default parameters
    char* prompt = "Once upon a time";
    float temperature = 0.0f;
    float topp = 0.9f;
    int steps = 200; // now: max *output* tokens
    unsigned long long seed = 12345;

    // Optional test name for benchmark output: llm "<prompt>" <steps> <temp> <top_p> <seed> <test_name>
    if (argc >= 2) prompt = argv[1];
    if (argc >= 3) steps = xv6_atoi(argv[2]);
    if (argc >= 4) temperature = xv6_atof(argv[3]);
    if (argc >= 5) topp = xv6_atof(argv[4]);
    if (argc >= 6) seed = xv6_atoi(argv[5]);
    if (argc >= 7) g_test_name = argv[6];  // e.g. "T1", "T2", ...

    printf(stdout,"prompt: %s\n",prompt);
    printf(stdout,"steps (max output tokens): %d\nTemperature %f\n",steps,temperature);
    printf(stdout,"Topp_n%f\nSeed %lld\n",topp,seed);
    printf(stdout,"Initializing LLM in xV6...\n");
    
    // Initialize transformer
    Transformer transformer;
    
    // Load model configuration and weights
    printf(stdout,"Loading model weights...\n");
    read_checkpoint(&transformer.config, &transformer.weights, 
                    &transformer.memory, &transformer.memory_size);
    
    // Allocate run state
    printf(stdout,"Allocating run state...\n");
    malloc_run_state(&transformer.state, &transformer.config);
    
    // Load tokenizer
    printf(stdout,"Loading tokenizer...\n");
    Tokenizer* tokenizer = build_tokenizer(transformer.config.vocab_size);
    
    printf(stdout,"Starting generation with prompt: %s\n", prompt);
    printf(stdout,"Steps (max output tokens): %d, Temperature: %f, Topp: %f Seed: %llu\n", 
           steps, temperature, topp, seed);

    // Reset hotspots for this run
    reset_benchmark_counters();
    
    // Run generation (now benchmark-aware)
    generate(&transformer, tokenizer, prompt, steps, temperature, topp, seed);
    
    // Cleanup
    free_tokenizer(tokenizer, transformer.config.vocab_size);
    free_transformer(&transformer);
    
    return 0;
}
