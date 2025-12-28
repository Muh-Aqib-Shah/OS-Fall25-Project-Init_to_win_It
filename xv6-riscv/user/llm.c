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
#define memcpy xv6_memcpy
#define memset xv6_memset
#define strcpy xv6_strcpy
#define strcmp xv6_strcmp
#define strcat xv6_strcat

// Common macros
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

// File descriptors in xv6
#define STDERR_FD 2

const char* g_test_name = "RUN";

#define NUM_THREADS 4

// Timing variables for benchmarking
unsigned long long matmul_cycles = 0;
unsigned long long activation_cycles = 0;
unsigned long long attention_cycles = 0;
unsigned long long ffn_cycles = 0;
unsigned long long sampling_cycles = 0;

#define xprintf(fd, fmt, ...) do { \
    if ((fd) == STDERR_FD) fprintf(fd, "Error: " fmt, ##__VA_ARGS__); \
    else printf(fmt, ##__VA_ARGS__); \
} while (0)

// Model configuration
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

// Transformer weights
typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* freq_cis_real;
    float* freq_cis_imag;
    float* wcls;
} TransformerWeights;

// Run state
typedef struct {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

// Transformer model
typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float* memory;
    int memory_size;
} Transformer;

#define CPU_FREQ_HZ 10000000.0
unsigned long long g_program_start_cycles = 0;

void reset_benchmark_counters(void) {
    matmul_cycles = 0;
    activation_cycles = 0;
    attention_cycles = 0;
    ffn_cycles = 0;
    sampling_cycles = 0;
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    unsigned long long start = gettime();
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
    activation_cycles += (gettime() - start);
}

void softmax(float* x, int size) {
    unsigned long long start = gettime();
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
    activation_cycles += (gettime() - start);
}

void matmul_parallel(float* output, float* input, float* weight, int n, int d);
void qkv_parallel(float* q, float* k, float* v, float* xb,
                  float* wq, float* wk, float* wv,
                  int dim, int n_heads, int head_size);
void attention_parallel(RunState* s, int n_heads, int head_size,
                        int seq_len, int pos, int loff, int dim);

void matmul(float* output, float* input, float* weight, int n, int d) {
    matmul_parallel(output, input, weight, n, d);
}

typedef struct {
    float* output;
    float* input;
    float* weight;
    int n;
    int d;
    int row_start;
    int row_end;
} MatmulArgs;

void matmul_worker(void *arg) {
    MatmulArgs* args = (MatmulArgs*)arg;
    for (int i = args->row_start; i < args->row_end; i++) {
        float val = 0.0f;
        for (int j = 0; j < args->n; j++) {
            val += args->weight[i * args->n + j] * args->input[j];
        }
        args->output[i] = val;
    }
    thread_exit();
}

void matmul_parallel(float* output, float* input, float* weight, int n, int d) {
    if (d < NUM_THREADS * 4) {
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) val += weight[i * n + j] * input[j];
            output[i] = val;
        }
        return;
    }

    int *thread_ids = malloc(NUM_THREADS * sizeof(int));
    MatmulArgs *args = malloc(NUM_THREADS * sizeof(MatmulArgs));
    if (!thread_ids || !args) {
        if (thread_ids) free(thread_ids);
        if (args) free(args);
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) val += weight[i * n + j] * input[j];
            output[i] = val;
        }
        return;
    }

    int rows_per_thread = d / NUM_THREADS;
    int extra_rows = d % NUM_THREADS;
    int current_row = 0;
    int created_count = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int rows_for_this_thread = rows_per_thread + (t < extra_rows ? 1 : 0);
        args[t].output = output;
        args[t].input = input;
        args[t].weight = weight;
        args[t].n = n;
        args[t].d = d;
        args[t].row_start = current_row;
        args[t].row_end = current_row + rows_for_this_thread;
        thread_ids[t] = thread_create(matmul_worker, &args[t]);
        if (thread_ids[t] < 0) {
            for (int j = 0; j < created_count; j++) thread_join(thread_ids[j]);
            for (int i = current_row; i < d; i++) {
                float val = 0.0f;
                for (int j = 0; j < n; j++) val += weight[i * n + j] * input[j];
                output[i] = val;
            }
            free(thread_ids);
            free(args);
            return;
        }
        created_count++;
        current_row += rows_for_this_thread;
    }
    for (int t = 0; t < NUM_THREADS; t++) thread_join(thread_ids[t]);
    free(thread_ids);
    free(args);
}

typedef struct {
    float* q;
    float* k;
    float* v;
    float* xb;
    float* wq;
    float* wk;
    float* wv;
    int dim;
    int head_start;
    int head_end;
    int head_size;
} QKVArgs;

void qkv_worker(void *arg) {
    QKVArgs* args = (QKVArgs*)arg;
    for (int h = args->head_start; h < args->head_end; h++) {
        int offset = h * args->head_size;
        for (int i = 0; i < args->head_size; i++) {
            float val = 0.0f;
            for (int j = 0; j < args->dim; j++) {
                val += args->wq[(offset + i) * args->dim + j] * args->xb[j];
            }
            args->q[offset + i] = val;
        }
        for (int i = 0; i < args->head_size; i++) {
            float val = 0.0f;
            for (int j = 0; j < args->dim; j++) {
                val += args->wk[(offset + i) * args->dim + j] * args->xb[j];
            }
            args->k[offset + i] = val;
        }
        for (int i = 0; i < args->head_size; i++) {
            float val = 0.0f;
            for (int j = 0; j < args->dim; j++) {
                val += args->wv[(offset + i) * args->dim + j] * args->xb[j];
            }
            args->v[offset + i] = val;
        }
    }
    thread_exit();
}

void qkv_parallel(float* q, float* k, float* v, float* xb,
                  float* wq, float* wk, float* wv,
                  int dim, int n_heads, int head_size) {
    if (n_heads < NUM_THREADS) {
        for (int h = 0; h < n_heads; h++) {
            int offset = h * head_size;
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++) {
                    val += wq[(offset + i) * dim + j] * xb[j];
                }
                q[offset + i] = val;
            }
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++) {
                    val += wk[(offset + i) * dim + j] * xb[j];
                }
                k[offset + i] = val;
            }
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++) {
                    val += wv[(offset + i) * dim + j] * xb[j];
                }
                v[offset + i] = val;
            }
        }
        return;
    }

    int *thread_ids = malloc(NUM_THREADS * sizeof(int));
    QKVArgs *args = malloc(NUM_THREADS * sizeof(QKVArgs));
    if (!thread_ids || !args) {
        if (thread_ids) free(thread_ids);
        if (args) free(args);
        for (int h = 0; h < n_heads; h++) {
            int offset = h * head_size;
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++)
                    val += wq[(offset + i) * dim + j] * xb[j];
                q[offset + i] = val;
            }
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++)
                    val += wk[(offset + i) * dim + j] * xb[j];
                k[offset + i] = val;
            }
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int j = 0; j < dim; j++)
                    val += wv[(offset + i) * dim + j] * xb[j];
                v[offset + i] = val;
            }
        }
        return;
    }

    int heads_per_thread = n_heads / NUM_THREADS;
    int extra_heads = n_heads % NUM_THREADS;
    int current_head = 0;

    for (int t = 0; t < NUM_THREADS; t++) {
        int heads_for_this_thread = heads_per_thread + (t < extra_heads ? 1 : 0);
        args[t].q = q;
        args[t].k = k;
        args[t].v = v;
        args[t].xb = xb;
        args[t].wq = wq;
        args[t].wk = wk;
        args[t].wv = wv;
        args[t].dim = dim;
        args[t].head_start = current_head;
        args[t].head_end = current_head + heads_for_this_thread;
        args[t].head_size = head_size;
        thread_ids[t] = thread_create(qkv_worker, &args[t]);
        current_head += heads_for_this_thread;
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        if (thread_ids[t] >= 0) {
            thread_join(thread_ids[t]);
        }
    }
    free(thread_ids);
    free(args);
}

typedef struct {
    float* att;
    float* q;
    float* key_cache;
    float* xb;
    float* value_cache;
    int seq_len;
    int head_size;
    int pos;
    int loff;
    int dim;
    int head_start;
    int head_end;
} AttentionArgs;

void attention_worker(void *arg) {
    AttentionArgs* args = (AttentionArgs*)arg;
    for (int h = args->head_start; h < args->head_end; h++) {
        float* q = args->q + h * args->head_size;
        float* att = args->att + h * args->seq_len;
        for (int t = 0; t <= args->pos; t++) {
            float* k = &(args->key_cache[args->loff + t * args->dim + h * args->head_size]);
            float score = 0.0f;
            for (int i = 0; i < args->head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(args->head_size);
            att[t] = score;
        }
        float max_val = att[0];
        for (int t = 1; t <= args->pos; t++) {
            if (att[t] > max_val) max_val = att[t];
        }
        float sum = 0.0f;
        for (int t = 0; t <= args->pos; t++) {
            att[t] = expf(att[t] - max_val);
            sum += att[t];
        }
        for (int t = 0; t <= args->pos; t++) {
            att[t] /= sum;
        }
        float* xb = args->xb + h * args->head_size;
        memset(xb, 0, args->head_size * sizeof(float));
        for (int t = 0; t <= args->pos; t++) {
            float* v = &(args->value_cache[args->loff + t * args->dim + h * args->head_size]);
            float a = att[t];
            for (int i = 0; i < args->head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
    thread_exit();
}

void attention_parallel(RunState* s, int n_heads, int head_size,
                       int seq_len, int pos, int loff, int dim) {
    if (n_heads < NUM_THREADS) {
        for (int h = 0; h < n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = &(s->key_cache[loff + t * dim + h * head_size]);
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            float max_val = att[0];
            for (int t = 1; t <= pos; t++) {
                if (att[t] > max_val) max_val = att[t];
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                att[t] = expf(att[t] - max_val);
                sum += att[t];
            }
            for (int t = 0; t <= pos; t++) {
                att[t] /= sum;
            }
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = &(s->value_cache[loff + t * dim + h * head_size]);
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        return;
    }

    int *thread_ids = malloc(NUM_THREADS * sizeof(int));
    AttentionArgs *args = malloc(NUM_THREADS * sizeof(AttentionArgs));
    if (!thread_ids || !args) {
        if (thread_ids) free(thread_ids);
        if (args) free(args);
        for (int h = 0; h < n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = &(s->key_cache[loff + t * dim + h * head_size]);
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                    score += q[i] * k[i];
                att[t] = score / sqrtf(head_size);
            }
            float max_val = att[0];
            for (int t = 1; t <= pos; t++)
                if (att[t] > max_val) max_val = att[t];
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++)
                sum += (att[t] = expf(att[t] - max_val));
            for (int t = 0; t <= pos; t++)
                att[t] /= sum;

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = &(s->value_cache[loff + t * dim + h * head_size]);
                for (int i = 0; i < head_size; i++)
                    xb[i] += att[t] * v[i];
            }
        }
        return;
    }

    int heads_per_thread = n_heads / NUM_THREADS;
    int extra_heads = n_heads % NUM_THREADS;
    int current_head = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        int heads_for_this_thread = heads_per_thread + (t < extra_heads ? 1 : 0);
        args[t].att = s->att;
        args[t].q = s->q;
        args[t].key_cache = s->key_cache;
        args[t].xb = s->xb;
        args[t].value_cache = s->value_cache;
        args[t].seq_len = seq_len;
        args[t].head_size = head_size;
        args[t].pos = pos;
        args[t].loff = loff;
        args[t].dim = dim;
        args[t].head_start = current_head;
        args[t].head_end = current_head + heads_for_this_thread;
        thread_ids[t] = thread_create(attention_worker, &args[t]);
        current_head += heads_for_this_thread;
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        if (thread_ids[t] >= 0) {
            thread_join(thread_ids[t]);
        }
    }
    free(thread_ids);
    free(args);
}

void transformer_forward(int token, int pos, Config* p, TransformerWeights* w, RunState* s) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(s->x, content_row, dim * sizeof(float));
    float* freq_cis_real_row = &(w->freq_cis_real[pos * head_size / 2]);
    float* freq_cis_imag_row = &(w->freq_cis_imag[pos * head_size / 2]);
    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, s->x, &(w->rms_att_weight[l * dim]), dim);
        unsigned long long att_start = gettime();
        unsigned long long matmul_start = gettime();
        qkv_parallel(s->q, s->k, s->v, s->xb,
                    &(w->wq[l * dim * dim]),
                    &(w->wk[l * dim * dim]),
                    &(w->wv[l * dim * dim]),
                    dim, p->n_heads, head_size);
        matmul_cycles += (gettime() - matmul_start);
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* k = s->k + h * head_size;
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
        int loff = l * p->seq_len * dim;
        float* key_cache_row   = &(s->key_cache[loff + pos * dim]);
        float* value_cache_row = &(s->value_cache[loff + pos * dim]);
        memcpy(key_cache_row,   s->k, dim * sizeof(float));
        memcpy(value_cache_row, s->v, dim * sizeof(float));
        attention_parallel(s, p->n_heads, head_size, p->seq_len, pos, loff, dim);
        matmul_start = gettime();
        matmul(s->xb2, s->xb, &(w->wo[l * dim * dim]), dim, dim);
        matmul_cycles += (gettime() - matmul_start);
        attention_cycles += (gettime() - att_start);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        rmsnorm(s->xb, s->x, &(w->rms_ffn_weight[l * dim]), dim);
        unsigned long long ffn_start = gettime();
        matmul_start = gettime();
        matmul(s->hb,  s->xb, &(w->w1[l * dim * hidden_dim]), dim, hidden_dim);
        matmul(s->hb2, s->xb, &(w->w3[l * dim * hidden_dim]), dim, hidden_dim);
        matmul_cycles += (gettime() - matmul_start);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul_start = gettime();
        matmul(s->xb, s->hb, &(w->w2[l * hidden_dim * dim]), hidden_dim, dim);
        matmul_cycles += (gettime() - matmul_start);
        ffn_cycles += (gettime() - ffn_start);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
    rmsnorm(s->x, s->x, w->rms_final_weight, dim);
    unsigned long long cls_start = gettime();
    matmul(s->logits, s->x, w->token_embedding_table, dim, p->vocab_size);
    matmul_cycles += (gettime() - cls_start);
}

int sample(float* probabilities, int n, float coin) {
    unsigned long long start = gettime();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            sampling_cycles += (gettime() - start);
            return i;
        }
    }
    sampling_cycles += (gettime() - start);
    return n - 1;
}

int cmp_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

int apply_top_p(float* probabilities, int n, float top_p) {
    float* sorted = (float*)malloc(n * sizeof(float));
    if (!sorted) return -1;
    memcpy(sorted, probabilities, n * sizeof(float));
    xv6_qsort(sorted, n, sizeof(float), cmp_desc);
    float cum_prob = 0.0f;
    int cutoff = 0;
    for (int i = 0; i < n; i++) {
        cum_prob += sorted[i];
        cutoff = i;
        if (cum_prob >= top_p) break;
    }
    float threshold = sorted[cutoff];
    for (int i = 0; i < n; i++) {
        if (probabilities[i] < threshold) probabilities[i] = 0.0f;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probabilities[i];
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) probabilities[i] /= sum;
    }
    free(sorted);
    return 0;
}

int argmax(float* v, int n) {
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
        printf("Failed to load tokenizer\n");
        exit(1);
    }
    Tokenizer* t = (Tokenizer*)malloc(sizeof(Tokenizer));
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    char* ptr = tokenizer_data;
    memcpy(&t->max_token_length, ptr, sizeof(int));
    ptr += sizeof(int);
    for (int i = 0; i < vocab_size; i++) {
        memcpy(&t->vocab_scores[i], ptr, sizeof(float));
        ptr += sizeof(float);
        int len;
        memcpy(&len, ptr, sizeof(int));
        ptr += sizeof(int);
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
    if (tokenizer->vocab) {
        for (int i = 0; i < vocab_size; i++) {
            if (tokenizer->vocab[i]) {
                free(tokenizer->vocab[i]);
            }
        }
        free(tokenizer->vocab);
    }
    if (tokenizer->vocab_scores) {
        free(tokenizer->vocab_scores);
    }
    free(tokenizer);
}

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
    ptr += p->seq_len * head_size / 2;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(Config* config, TransformerWeights* weights,
                     float** memory, int* memory_size) {
    int size_bytes = 0;
    char* raw_weights = fetch_model_weights(&size_bytes);
    if (!raw_weights) {
        fprintf(STDERR_FD,"Failed to fetch model weights from server\n");
        exit(1);
    }
    if (size_bytes < sizeof(Config)) {
        fprintf(STDERR_FD,"Model file too small to contain Config struct\n");
        free(raw_weights);
        exit(1);
    }
    memcpy(config, raw_weights, sizeof(Config));
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = fabs(config->vocab_size);
    int dim        = config->dim;
    int hidden_dim = config->hidden_dim;
    int n_layers   = config->n_layers;
    int n_heads    = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int vocab_size = config->vocab_size;
    int seq_len    = config->seq_len;
    int head_size  = dim / n_heads;
    printf("Loaded model config:\n");
    printf("  dim=%d, hidden_dim=%d, n_layers=%d\n", dim, hidden_dim, n_layers);
    printf("  n_heads=%d, n_kv_heads=%d, vocab_size=%d, seq_len=%d\n",
           n_heads, n_kv_heads, vocab_size, seq_len);
    printf("  shared_weights=%d\n", shared_weights);
    unsigned long long total_floats = 0;
    total_floats += (unsigned long long)vocab_size * dim;
    total_floats += (unsigned long long)n_layers * dim;
    total_floats += (unsigned long long)n_layers * dim * (n_heads * head_size);
    total_floats += (unsigned long long)n_layers * dim * (n_kv_heads * head_size);
    total_floats += (unsigned long long)n_layers * dim * (n_kv_heads * head_size);
    total_floats += (unsigned long long)n_layers * (n_heads * head_size) * dim;
    total_floats += (unsigned long long)n_layers * dim;
    total_floats += (unsigned long long)n_layers * dim * hidden_dim;
    total_floats += (unsigned long long)n_layers * hidden_dim * dim;
    total_floats += (unsigned long long)n_layers * dim * hidden_dim;
    total_floats += (unsigned long long)dim;
    total_floats += (unsigned long long)seq_len * head_size / 2;
    total_floats += (unsigned long long)seq_len * head_size / 2;
    if (!shared_weights) {
        total_floats += (unsigned long long)vocab_size * dim;
    }
    *memory_size = (int)total_floats;
    printf("Calculated weights size: %d floats (%llu bytes)\n",
           *memory_size, (unsigned long long)*memory_size * sizeof(float));
    unsigned long long expected_bytes = sizeof(Config) + total_floats * sizeof(float);
    printf("Expected file size: %llu bytes\n", expected_bytes);
    printf("Actual file size: %d bytes\n", size_bytes);
    if (fabs(size_bytes - expected_bytes) > 1024) {
        fprintf(STDERR_FD,"File Size mismatch! Expected %llu, got %d\n",
               expected_bytes, size_bytes);
        free(raw_weights);
        exit(1);
    }
    *memory = (float*)malloc(*memory_size * sizeof(float));
    if (!*memory) {
        fprintf(STDERR_FD,"malloc failed for model weights (%d floats)\n", *memory_size);
        free(raw_weights);
        exit(1);
    }
    memcpy(*memory, raw_weights + sizeof(Config), *memory_size * sizeof(float));
    free(raw_weights);
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
    printf("Model weights loaded successfully into memory\n");
}

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_heads = p->n_heads;
    s->x  = malloc(dim * sizeof(float));
    s->xb = malloc(dim * sizeof(float));
    s->xb2 = malloc(dim * sizeof(float));
    s->hb = malloc(hidden_dim * sizeof(float));
    s->hb2 = malloc(hidden_dim * sizeof(float));
    s->q = malloc(dim * sizeof(float));
    s->k = malloc(dim * sizeof(float));
    s->v = malloc(dim * sizeof(float));
    s->att = malloc(n_heads * p->seq_len * sizeof(float));
    s->logits = malloc(p->vocab_size * sizeof(float));
    s->key_cache = malloc(p->n_layers * p->seq_len * dim * sizeof(float));
    s->value_cache = malloc(p->n_layers * p->seq_len * dim * sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v || !s->att ||
        !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(STDERR_FD, "malloc failed!\n");
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

void generate(Transformer* model, Tokenizer* t, char* prompt,
              int steps, float temperature, float topp, unsigned long long seed) {
    Config* p = &model->config;
    RunState* s = &model->state;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    prompt_tokens[num_prompt_tokens++] = 1;
    for (int i = 0; prompt[i] != '\0'; i++) {
        prompt_tokens[num_prompt_tokens++] = (unsigned char)prompt[i] + 3;
    }
    int token = prompt_tokens[0];
    int pos = 0;
    int generated_tokens = 0;
    int output_tokens = 0;
    int ttft_measured = 0;
    unsigned long long ttft_cycles = 0;
    unsigned long long gen_start_cycles = 0;
    unsigned long long infer_start_cycles = gettime();
    (void)infer_start_cycles;
    printf("<start>\n");
    while (pos < p->seq_len && generated_tokens < (steps + num_prompt_tokens)) {
        transformer_forward(token, pos, p, &model->weights, s);
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            if (temperature == 0.0f) {
                next = argmax(s->logits, p->vocab_size);
            } else {
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
                float coin = (float)((seed >> 16) % 10000) / 10000.0f;
                seed = seed * 1664525ULL + 1013904223ULL;
                next = sample(s->logits, p->vocab_size, coin);
            }
            if (!ttft_measured) {
                unsigned long long first_token_cycle = gettime();
                ttft_cycles = first_token_cycle - g_program_start_cycles;
                gen_start_cycles = first_token_cycle;
                ttft_measured = 1;
            }
            output_tokens++;
        }
        if (next == 2) {
            break;
        } else if (next < 3) {
        } else if (next < 259) {
            printf("%c", (char)(next - 3));
            generated_tokens++;
        } else if (next < t->vocab_size && t->vocab[next]) {
            printf("%s", t->vocab[next]);
            generated_tokens++;
        }
        token = next;
        pos++;
    }
    printf("\n<end>\n");
    unsigned long long end_cycles = gettime();
    unsigned long long e2e_cycles = end_cycles - g_program_start_cycles;
    if (gen_start_cycles == 0) {
        gen_start_cycles = end_cycles;
    }
    unsigned long long gen_cycles = end_cycles - gen_start_cycles;
    double ttft_seconds = (double)ttft_cycles / CPU_FREQ_HZ;
    double e2e_seconds  = (double)e2e_cycles / CPU_FREQ_HZ;
    double tps = 0.0;
    if (output_tokens > 1 && gen_cycles > 0) {
        double gen_seconds = (double)gen_cycles / CPU_FREQ_HZ;
        tps = (double)(generated_tokens - 1) / gen_seconds;
    }
    unsigned long long hotspot_total =
        matmul_cycles + activation_cycles + attention_cycles +
        ffn_cycles + sampling_cycles;
    if (hotspot_total == 0) hotspot_total = 1;
    double matmul_pct     = 100.0 * (double)matmul_cycles     / (double)hotspot_total;
    double activation_pct = 100.0 * (double)activation_cycles / (double)hotspot_total;
    double attention_pct  = 100.0 * (double)attention_cycles  / (double)hotspot_total;
    double ffn_pct        = 100.0 * (double)ffn_cycles        / (double)hotspot_total;
    double sampling_pct   = 100.0 * (double)sampling_cycles   / (double)hotspot_total;
    int ttft_sec_int  = (int)ttft_seconds;
    int ttft_sec_frac = (int)((ttft_seconds - ttft_sec_int) * 1000000.0 + 0.5);
    if (ttft_sec_frac < 0) ttft_sec_frac = -ttft_sec_frac;
    int e2e_sec_int  = (int)e2e_seconds;
    int e2e_sec_frac = (int)((e2e_seconds - e2e_sec_int) * 1000000.0 + 0.5);
    if (e2e_sec_frac < 0) e2e_sec_frac = -e2e_sec_frac;
    int tps_int  = (int)tps;
    int tps_frac = (int)((tps - tps_int) * 1000.0 + 0.5);
    if (tps_frac < 0) tps_frac = -tps_frac;
    int matmul_int  = (int)matmul_pct;
    int matmul_frac = (int)((matmul_pct - matmul_int) * 10.0 + 0.5);
    int act_int     = (int)activation_pct;
    int act_frac    = (int)((activation_pct - act_int) * 10.0 + 0.5);
    int att_int     = (int)attention_pct;
    int att_frac    = (int)((attention_pct - att_int) * 10.0 + 0.5);
    int ffn_int     = (int)ffn_pct;
    int ffn_frac    = (int)((ffn_pct - ffn_int) * 10.0 + 0.5);
    int samp_int    = (int)sampling_pct;
    int samp_frac   = (int)((sampling_pct - samp_int) * 10.0 + 0.5);
    if (matmul_frac < 0) matmul_frac = -matmul_frac;
    if (act_frac < 0)    act_frac    = -act_frac;
    if (att_frac < 0)    att_frac    = -att_frac;
    if (ffn_frac < 0)    ffn_frac    = -ffn_frac;
    if (samp_frac < 0)   samp_frac   = -samp_frac;
    extern const char* g_test_name;
    printf( "\n=== TEST RUN %s ===\n", g_test_name);
    printf( "Prompt:\n\"%s\"\n", prompt);
    printf( "Prompt Tokens: %d\n", num_prompt_tokens);
    printf( "Output Tokens: %d\n", output_tokens);
    printf( "Total Generated Tokens: %d\n", generated_tokens);
    printf( "Temperature: %d.%d\n", (int)temperature, 0);
    printf( "Seed: %llu\n", seed);
    printf( "PRIMARY METRICS:\n");
    printf( "TTFT: %llu cycles (%d.%d seconds)\n",
       ttft_cycles, ttft_sec_int, ttft_sec_frac);
    printf( "TPS: %d.%d tokens/sec\n", tps_int, tps_frac);
    printf( "End-to-End: %llu cycles (%d.%d seconds)\n",
       e2e_cycles, e2e_sec_int, e2e_sec_frac);
    printf( "HOTSPOT BREAKDOWN (%% of inference time):\n");
    printf( "matmul(): %d.%d%%\n", matmul_int, matmul_frac);
    printf( "Activations (expf, sqrtf): %d.%d%%\n", act_int, act_frac);
    printf( "Attention: %d.%d%%\n", att_int, att_frac);
    printf( "FFN: %d.%d%%\n", ffn_int, ffn_frac);
    printf( "Sampling: %d.%d%%\n", samp_int, samp_frac);
    printf( "NOTES: Categories add to 100%%. matmul time counted separately.\n");
    printf( "===================\n");
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {
    g_program_start_cycles = gettime();
    char* prompt = "Once upon a time";
    float temperature = 0.0f;
    float topp = 1.0f;
    int steps = 100;
    unsigned long long seed = 12345;
    if (argc >= 2) prompt = argv[1];
    if (argc >= 3) steps = xv6_atoi(argv[2]);
    if (argc >= 4) temperature = xv6_atof(argv[3]);
    if (argc >= 5) topp = xv6_atof(argv[4]);
    if (argc >= 6) seed = xv6_atoi(argv[5]);
    if (argc >= 7) g_test_name = argv[6];
    printf( "DEBUG: argc = %d\n", argc);
    printf( "DEBUG: prompt = %s\n", prompt);
    printf( "DEBUG: steps = %d\n", steps);
    printf( "DEBUG: temperature = %d.%d\n", (int)temperature, 0);
    printf( "DEBUG: topp = %d.%d\n", (int)topp, 0);
    printf( "DEBUG: seed = %llu\n", seed);
    printf( "DEBUG: test_name = %s\n", g_test_name);
    printf( "prompt: %s\n", prompt);
    printf( "steps (max output tokens): %d\nTemperature %d.%d\n",
           steps, (int)temperature, 0);
    printf( "prompt: %s\n", prompt);
    printf( "steps (max output tokens): %d\nTemperature %d.%d\n",
           steps, (int)temperature, 0);
    printf( "Topp_n%d.%d\n", (int)topp, 0);
    printf( "Seed %llu\n", seed);
    printf( "Initializing LLM in xV6...\n");
    Transformer transformer;
    printf("Loading model weights...\n");
    read_checkpoint(&transformer.config, &transformer.weights,
                    &transformer.memory, &transformer.memory_size);
    printf("Allocating run state...\n");
    malloc_run_state(&transformer.state, &transformer.config);
    printf("Loading tokenizer...\n");
    Tokenizer* tokenizer = build_tokenizer(transformer.config.vocab_size);
    printf( "Starting generation with prompt: %s\n", prompt);
    printf( "Steps (max output tokens): %d, Temperature: %d.%d, Topp: %d.%d Seed: %llu\n",
           steps, (int)temperature, 0, (int)topp, 0, seed);
    reset_benchmark_counters();
    generate(&transformer, tokenizer, prompt, steps, temperature, topp, seed);
    free_tokenizer(tokenizer, transformer.config.vocab_size);
    free_transformer(&transformer);
    return 0;
}
