import cv2
import time

DROIDCAM_URL = None  # Set to e.g. "http://192.168.1.100:4747/video" to test URL stream

def test_cameras():
    print("Testing to find available cameras...")
    available_cameras = []
    
    # Check first 5 camera indices
    for i in range(5):
        print(f"Testing camera index {i}...")
        
        # Try without CAP_DSHOW first
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  [SUCCESS] Camera index {i} successfully opened (default backend).")
                # Check if frame is totally black
                if frame.max() == 0:
                    print("    -> WARNING: The frame is completely black! It might be a virtual camera (like OBS) or has a physical privacy cover.")
                else:
                    print("    -> Frame looks good (has some content/color).")
                
                available_cameras.append((i, "default"))
            cap.release()
            
        # Try with CAP_DSHOW if Windows
        cap_dshow = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap_dshow.isOpened():
            ret, frame = cap_dshow.read()
            if ret:
                print(f"  [SUCCESS] Camera index {i} successfully opened (CAP_DSHOW).")
                if frame.max() == 0:
                    print("    -> WARNING: The frame is completely black!")
                available_cameras.append((i, "DSHOW"))
            cap_dshow.release()

    if not available_cameras:
        print("\nNo working cameras found! Is your camera plugged in and accessible?")
        return

    print("\n--- Testing Display ---")
    print("Press 'q' in the window to close it and move to the next camera if any.")
    
    for idx, backend in available_cameras:
        print(f"\nOpening Camera {idx} ({backend})...")
        if backend == "DSHOW":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)
            
        if not cap.isOpened():
            continue
            
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            cv2.imshow(f"Camera {idx} ({backend} backend)", frame)
            
            # Use 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def test_url_stream(url: str):
    print(f"\n--- Testing URL Stream: {url} ---")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("  [FAILED] Could not open URL stream. Check IP address and that DroidCam is running.")
        return
    ret, frame = cap.read()
    if not ret or frame is None:
        print("  [FAILED] Opened but could not read frame from URL.")
        cap.release()
        return
    print(f"  [SUCCESS] URL stream opened! Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("  Press 'q' in the preview window to close.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("DroidCam URL Stream Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_cameras()
    if DROIDCAM_URL:
        test_url_stream(DROIDCAM_URL)
    else:
        print("\n--- URL Stream Test ---")
        print("To test DroidCam URL stream, edit test_camera.py and set DROIDCAM_URL at the top.")
        print('Example: DROIDCAM_URL = "http://192.168.1.100:4747/video"')