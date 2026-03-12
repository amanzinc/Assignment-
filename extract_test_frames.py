import cv2
import os

def main():
    video_path = 'Dataset/Split_A/split_a_barn.mp4'
    output_dir = 'test_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    stride = 10
    for i in range(5):
        frame_idx = i * stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            fname = os.path.join(output_dir, f"frame_{i:03d}.png")
            cv2.imwrite(fname, frame)
            print(f"Saved {fname}")
        else:
            print(f"Failed to read frame {frame_idx}")

    cap.release()

if __name__ == "__main__":
    main()
