# import cv2
# # vid = cv2.VideoCapture(0) # For webcam
# vid = cv2.VideoCapture("http://88.131.30.164/mjpg/video.mjpg") # For streaming links
# while True:
#   _,frame = vid.read()
#   print(frame)
#   cv2.imshow('Video Live IP cam',frame)
#   key = cv2.waitKey(1) & 0xFF
#   if key ==ord('q'):
#     break

# vid.release()
# cv2.destroyAllWindows()

import socket
import threading
import time

# Define the IP addresses, ports, and IP prefixes to block
blocked_ips = ['10.0.0.1', '192.168.1.1']
blocked_ports = [80, 443]
blocked_prefixes = ['192.168.1.']
blocked_requests = {}

# Define the function to block incoming connections
def block_connection(conn, addr):
    # Check if the IP address is blocked
    if addr[0] in blocked_ips:
        print(f"Blocked connection from {addr[0]}")
        conn.close()
        return

    # Check if the port is blocked
    if addr[1] in blocked_ports:
        print(f"Blocked connection to port {addr[1]} from {addr[0]}")
        conn.close()
        return

    # Check if the IP prefix is blocked
    for prefix in blocked_prefixes:
        if addr[0].startswith(prefix):
            print(f"Blocked connection from {addr[0]} due to blocked prefix {prefix}")
            conn.close()
            return

    # Check if the IP has made too many requests in a short period of time
    if addr[0] in blocked_requests:
        if time.time() - blocked_requests[addr[0]] < 60:
            print(f"Blocked connection from {addr[0]} due to too many requests in a short period of time")
            conn.close()
            return
    else:
        blocked_requests[addr[0]] = time.time()

    # If the connection is allowed, process the request
    print(f"Processing request from {addr[0]}")
    # Process the request here
    conn.close()

# Define the main function to listen for incoming connections
def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 80))
    server.listen(5)
    print("Firewall program running...")

    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=block_connection, args=(conn, addr))
        t.start()

if __name__ == '__main__':
    main()
