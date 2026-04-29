import cv2
import os

class VideoProcessor:
    def __init__(self):
        self.output_dir = os.path.join("static", "videos")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_highlight(self, video_path, start_frame, end_frame, output_name):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.output_dir, output_name)
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1
            
        cap.release()
        out.release()
        print(f"Highlight clip saved: {out_path}")
        return out_path
