#!/usr/bin/env python3
"""
Simple test client for the file server
Run this AFTER starting file_server.py in another terminal
"""

import socket
import struct

SERVER_IP = "127.0.0.1"
SERVER_PORT = 9999

# Message types
MSG_METADATA_REQUEST  = 0x01
MSG_METADATA_RESPONSE = 0x02
MSG_DATA_REQUEST      = 0x03
MSG_DATA_RESPONSE     = 0x04
MSG_ERROR             = 0xFF

def test_metadata_request(file_id):
    """Test metadata request"""
    print(f"\n{'='*50}")
    print(f"Testing METADATA_REQUEST for file {file_id}")
    print(f"{'='*50}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)
    
    # Build request: type(1) + file_id(1)
    request = struct.pack('!BB', MSG_METADATA_REQUEST, file_id)
    
    try:
        sock.sendto(request, (SERVER_IP, SERVER_PORT))
        print(f"Sent: METADATA_REQUEST for file {file_id}")
        
        response, addr = sock.recvfrom(4096)
        print(f"Received {len(response)} bytes from {addr}")
        
        if response[0] == MSG_METADATA_RESPONSE:
            msg_type, size, packet_count = struct.unpack('!BII', response[:9])
            sha256 = response[9:41].hex()
            
            print(f"\nMETADATA_RESPONSE:")
            print(f"  File size: {size:,} bytes")
            print(f"  Packet count: {packet_count}")
            print(f"  SHA-256: {sha256}")
            return True
        elif response[0] == MSG_ERROR:
            print(f"ERROR response: error code 0x{response[1]:02x}")
            return False
        else:
            print(f"Unexpected response type: 0x{response[0]:02x}")
            return False
            
    except socket.timeout:
        print("TIMEOUT - No response received")
        print("Make sure file_server.py is running!")
        return False
    finally:
        sock.close()

def test_data_request(file_id, packet_idx):
    """Test data request"""
    print(f"\n{'='*50}")
    print(f"Testing DATA_REQUEST for file {file_id}, packet {packet_idx}")
    print(f"{'='*50}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)
    
    # Build request: type(1) + file_id(1) + packet_idx(4)
    request = struct.pack('!BBI', MSG_DATA_REQUEST, file_id, packet_idx)
    
    try:
        sock.sendto(request, (SERVER_IP, SERVER_PORT))
        print(f"Sent: DATA_REQUEST for file {file_id}, packet {packet_idx}")
        
        response, addr = sock.recvfrom(4096)
        print(f"Received {len(response)} bytes from {addr}")
        
        if response[0] == MSG_DATA_RESPONSE:
            msg_type, recv_idx, data_len = struct.unpack('!BIH', response[:7])
            data = response[7:]
            
            print(f"\nDATA_RESPONSE:")
            print(f"  Packet index: {recv_idx}")
            print(f"  Data length: {data_len}")
            print(f"  Actual data received: {len(data)} bytes")
            print(f"  First 20 bytes (hex): {data[:20].hex()}")
            return True
        elif response[0] == MSG_ERROR:
            print(f"ERROR response: error code 0x{response[1]:02x}")
            return False
        else:
            print(f"Unexpected response type: 0x{response[0]:02x}")
            return False
            
    except socket.timeout:
        print("TIMEOUT - No response received")
        print("Make sure file_server.py is running!")
        return False
    finally:
        sock.close()

def test_invalid_file():
    """Test requesting invalid file ID"""
    print(f"\n{'='*50}")
    print(f"Testing invalid file ID (should return error)")
    print(f"{'='*50}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)
    
    # Request file ID 99 (doesn't exist)
    request = struct.pack('!BB', MSG_METADATA_REQUEST, 99)
    
    try:
        sock.sendto(request, (SERVER_IP, SERVER_PORT))
        print(f"Sent: METADATA_REQUEST for invalid file 99")
        
        response, addr = sock.recvfrom(4096)
        
        if response[0] == MSG_ERROR:
            print(f"Got expected ERROR response (error code: 0x{response[1]:02x})")
            return True
        else:
            print(f"Unexpected response type: 0x{response[0]:02x}")
            return False
            
    except socket.timeout:
        print("TIMEOUT")
        return False
    finally:
        sock.close()

def main():
    print("\n" + "="*60)
    print("   File Server Test Client")
    print("="*60)
    print("\nMake sure file_server.py is running in another terminal!\n")
    
    results = []
    
    # Test 1: Metadata for model (file 0)
    results.append(("Metadata - Model", test_metadata_request(0)))
    
    # Test 2: Metadata for tokenizer (file 1)
    results.append(("Metadata - Tokenizer", test_metadata_request(1)))
    
    # Test 3: Data request for first packet of model
    results.append(("Data - Packet 0", test_data_request(0, 0)))
    
    # Test 4: Data request for packet 100 of model
    results.append(("Data - Packet 100", test_data_request(0, 100)))
    
    # Test 5: Invalid file ID (should get error)
    results.append(("Invalid File ID", test_invalid_file()))
    
    # Summary
    print("\n" + "="*60)
    print("   TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed} passed, {failed} failed")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
