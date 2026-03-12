import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("output_dir", help="Directory to save frames")
    parser.add_argument("--stride", type=int, default=5, help="Frame extraction stride")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening {args.video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")
    
    extracted_count = 0
    for i in range(0, frame_count, args.stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            fname = os.path.join(args.output_dir, f"frame_{i:04d}.png")
            cv2.imwrite(fname, frame)
            extracted_count += 1
            if extracted_count % 50 == 0:
                print(f"Extracted {extracted_count} frames...")
        else:
            break

    print(f"Finished extracting {extracted_count} frames to {args.output_dir}")
    cap.release()

if __name__ == "__main__":
    main()
