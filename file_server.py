#!/usr/bin/env python3
"""
UDP File Server for LLM Weight Transfer
Implements RFTP (Reliable File Transfer Protocol)

Protocol Message Types:
  0x01 - METADATA_REQUEST: Client requests file info
  0x02 - METADATA_RESPONSE: Server sends file size, packet count, SHA-256
  0x03 - DATA_REQUEST: Client requests specific packet
  0x04 - DATA_RESPONSE: Server sends packet data
  0xFF - ERROR: Server indicates an error
"""

import socket
import struct
import hashlib
import sys
import os

# ============================================
# Configuration
# ============================================
SERVER_PORT = 9999
SERVER_IP = "0.0.0.0"  # Listen on all interfaces
PACKET_SIZE = 1024     # Bytes per data packet

# File paths (adjust these to your file locations)
MODEL_PATH = "stories15M.bin"
TOKENIZER_PATH = "tokenizer.bin"

# ============================================
# Message Types
# ============================================
MSG_METADATA_REQUEST  = 0x01
MSG_METADATA_RESPONSE = 0x02
MSG_DATA_REQUEST      = 0x03
MSG_DATA_RESPONSE     = 0x04
MSG_ERROR             = 0xFF

# Error Codes
ERR_INVALID_FILE_ID   = 0x01
ERR_INVALID_PACKET    = 0x02
ERR_INTERNAL          = 0x03

# File IDs
FILE_MODEL     = 0
FILE_TOKENIZER = 1

# ============================================
# File Server Class
# ============================================
class FileServer:
    def __init__(self, model_path, tokenizer_path):
        """Initialize server with file paths"""
        self.files = {}
        
        # Load files
        self.load_file(FILE_MODEL, model_path, "Model weights")
        self.load_file(FILE_TOKENIZER, tokenizer_path, "Tokenizer")
        
        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((SERVER_IP, SERVER_PORT))
        
        print(f"\n{'='*60}")
        print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
        print(f"{'='*60}\n")
    
    def load_file(self, file_id, path, name):
        """Load file into memory and compute metadata"""
        print(f"Loading {name} from: {path}")
        
        if not os.path.exists(path):
            print(f"  ERROR: File not found: {path}")
            print(f"  Please ensure the file exists in the correct location.")
            sys.exit(1)
        
        try:
            with open(path, 'rb') as f:
                data = f.read()
            
            # Compute SHA-256 checksum
            sha256 = hashlib.sha256(data).digest()
            sha256_hex = hashlib.sha256(data).hexdigest()
            
            # Calculate packet count
            packet_count = (len(data) + PACKET_SIZE - 1) // PACKET_SIZE
            
            self.files[file_id] = {
                'data': data,
                'size': len(data),
                'packet_count': packet_count,
                'sha256': sha256,
                'sha256_hex': sha256_hex,
                'name': name,
                'path': path
            }
            
            print(f"  Size: {len(data):,} bytes")
            print(f"  Packets: {packet_count}")
            print(f"  SHA-256: {sha256_hex}")
            print()
            
        except Exception as e:
            print(f"  ERROR loading file: {e}")
            sys.exit(1)
    
    def handle_metadata_request(self, data, addr):
        """Handle METADATA_REQUEST message"""
        if len(data) < 2:
            print(f"  Invalid request (too short)")
            return
        
        file_id = data[1]
        print(f"  File ID: {file_id}")
        
        if file_id not in self.files:
            # Send error response
            response = struct.pack('!BB', MSG_ERROR, ERR_INVALID_FILE_ID)
            self.sock.sendto(response, addr)
            print(f"  -> ERROR: Invalid file ID")
            return
        
        file_info = self.files[file_id]
        
        # Build response: type(1) + size(4) + packet_count(4) + sha256(32) = 41 bytes
        response = struct.pack('!BII', 
                              MSG_METADATA_RESPONSE,
                              file_info['size'],
                              file_info['packet_count'])
        response += file_info['sha256']
        
        self.sock.sendto(response, addr)
        print(f"  -> METADATA_RESPONSE: {file_info['size']} bytes, {file_info['packet_count']} packets")
    
    def handle_data_request(self, data, addr):
        """Handle DATA_REQUEST message"""
        if len(data) < 6:
            print(f"  Invalid request (too short)")
            return
        
        file_id = data[1]
        packet_idx = struct.unpack('!I', data[2:6])[0]
        
        if file_id not in self.files:
            response = struct.pack('!BB', MSG_ERROR, ERR_INVALID_FILE_ID)
            self.sock.sendto(response, addr)
            print(f"  -> ERROR: Invalid file ID")
            return
        
        file_info = self.files[file_id]
        
        if packet_idx >= file_info['packet_count']:
            response = struct.pack('!BB', MSG_ERROR, ERR_INVALID_PACKET)
            self.sock.sendto(response, addr)
            print(f"  -> ERROR: Invalid packet index {packet_idx}")
            return
        
        # Calculate offset and extract data
        offset = packet_idx * PACKET_SIZE
        end = min(offset + PACKET_SIZE, file_info['size'])
        packet_data = file_info['data'][offset:end]
        data_len = len(packet_data)
        
        # Build response: type(1) + packet_idx(4) + length(2) + data
        response = struct.pack('!BIH', MSG_DATA_RESPONSE, packet_idx, data_len)
        response += packet_data
        
        self.sock.sendto(response, addr)
        
        # Log progress (every 100 packets or first/last)
        if packet_idx % 100 == 0 or packet_idx == file_info['packet_count'] - 1:
            print(f"  -> DATA_RESPONSE: packet {packet_idx}/{file_info['packet_count']-1}, {data_len} bytes")
    
    def run(self):
        """Main server loop"""
        print("Waiting for requests...\n")
        
        request_count = 0
        
        try:
            while True:
                # Receive packet
                data, addr = self.sock.recvfrom(4096)
                request_count += 1
                
                if len(data) < 1:
                    continue
                
                msg_type = data[0]
                
                # Log request
                if msg_type == MSG_METADATA_REQUEST:
                    print(f"[{request_count}] METADATA_REQUEST from {addr}")
                    self.handle_metadata_request(data, addr)
                elif msg_type == MSG_DATA_REQUEST:
                    # Only log every 100th data request to reduce spam
                    file_id = data[1] if len(data) > 1 else -1
                    packet_idx = struct.unpack('!I', data[2:6])[0] if len(data) >= 6 else -1
                    if packet_idx % 100 == 0:
                        print(f"[{request_count}] DATA_REQUEST from {addr} - file {file_id}, packet {packet_idx}")
                    self.handle_data_request(data, addr)
                else:
                    print(f"[{request_count}] Unknown message type: 0x{msg_type:02x} from {addr}")
        
        except KeyboardInterrupt:
            print("\n\nServer shutting down...")
            self.sock.close()
            print("Goodbye!")

# ============================================
# Main Entry Point
# ============================================
def main():
    print()
    print("=" * 60)
    print("   LLM File Transfer Server (RFTP)")
    print("=" * 60)
    print()
    
    # File paths - can be overridden via command line
    model_path = MODEL_PATH
    tokenizer_path = TOKENIZER_PATH
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        tokenizer_path = sys.argv[2]
    
    print(f"Model file: {model_path}")
    print(f"Tokenizer file: {tokenizer_path}")
    print()
    
    # Start server
    server = FileServer(model_path, tokenizer_path)
    server.run()

if __name__ == "__main__":
    main()
