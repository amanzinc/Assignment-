import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("output_dir", help="Directory to save frames")
    parser.add_argument("--stride", type=int, default=5, help="Frame extraction stride")
    parser.add_argument("--max-width", type=int, default=0,
                        help="Resize frame to this max width before saving (0 = no resize). "
                             "Saves as JPEG when set, PNG otherwise.")
    parser.add_argument("--jpeg-quality", type=int, default=92,
                        help="JPEG quality 0-100 when --max-width is used (default 92)")
    args = parser.parse_args()

    use_jpeg = args.max_width > 0
    ext = ".jpg" if use_jpeg else ".png"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error opening {args.video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {frame_count}")
    
    write_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality] if use_jpeg else []

    # Sequential read: use cap.grab() to skip (no decode) and cap.retrieve() only on
    # wanted frames.  This avoids FFmpeg seek + decode-buffer bloat that OOMs on 4K video.
    extracted_count = 0
    for i in range(frame_count):
        if i % args.stride == 0:
            ret, frame = cap.read()
            if not ret:
                break
            if use_jpeg:
                h, w = frame.shape[:2]
                if w > args.max_width:
                    scale = args.max_width / float(w)
                    frame = cv2.resize(frame, (args.max_width, max(1, int(h * scale))),
                                       interpolation=cv2.INTER_AREA)
            fname = os.path.join(args.output_dir, f"frame_{i:04d}{ext}")
            cv2.imwrite(fname, frame, write_params)
            extracted_count += 1
            if extracted_count % 50 == 0:
                print(f"Extracted {extracted_count} frames...")
        else:
            if not cap.grab():
                break

    print(f"Finished extracting {extracted_count} frames to {args.output_dir}")
    cap.release()

if __name__ == "__main__":
    main()
