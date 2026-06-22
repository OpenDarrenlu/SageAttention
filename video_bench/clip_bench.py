#!/usr/bin/env python3
"""
评估生成视频与文本提示的对齐度
"""

import torch
import clip
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

class VideoTextAligner:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
    def extract_frames(self, video_path, num_frames=8):
        """均匀采样视频帧"""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total == 0:
            return []
        
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        current_idx = 0
        
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if i == indices[current_idx]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                current_idx += 1
                if current_idx >= len(indices):
                    break
        
        cap.release()
        return frames
    
    def encode_text(self, text):
        """编码文本"""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)
    
    def encode_image(self, image):
        """编码单张图像"""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        return features / features.norm(dim=-1, keepdim=True)
    
    def evaluate(self, video_path, text_prompt, num_frames=8):
        """
        评估视频与文本的对齐度
        
        返回：
        - text_alignment: 文本-视频平均相似度
        - temporal_consistency: 帧间一致性
        - frame_scores: 每帧的详细分数
        """
        frames = self.extract_frames(video_path, num_frames)
        if not frames:
            return None
        
        # 编码文本
        text_features = self.encode_text(text_prompt)
        
        # 编码所有帧
        frame_features = []
        frame_scores = []
        
        for frame in frames:
            feat = self.encode_image(frame)
            frame_features.append(feat)
            sim = (text_features @ feat.T).item()
            frame_scores.append(sim)
        
        # 计算指标
        text_alignment = np.mean(frame_scores)
        
        # 帧间一致性（相邻帧的相似度）
        if len(frame_features) > 1:
            consistencies = []
            for i in range(len(frame_features) - 1):
                sim = (frame_features[i] @ frame_features[i+1].T).item()
                consistencies.append(sim)
            temporal_consistency = np.mean(consistencies)
        else:
            temporal_consistency = 1.0
        
        # 帧间方差（反映闪烁程度）
        frame_variance = np.std(frame_scores)
        
        return {
            "text_alignment": float(text_alignment),
            "temporal_consistency": float(temporal_consistency),
            "frame_variance": float(frame_variance),
            "frame_scores": [float(s) for s in frame_scores],
            "num_frames": len(frames)
        }


# 使用示例
if __name__ == "__main__":
    # 初始化
    evaluator = VideoTextAligner(model_name="ViT-B/32")
    
    # 评估单个视频
    video_path = "generated_video.mp4"
    prompt = "a red car driving on a highway"
    
    result = evaluator.evaluate(video_path, prompt, num_frames=8)
    print(f"评估结果: {result}")
    
    # 批量评估
    video_dir = Path("./my_videos")
    results = {}
    
    for video_file in tqdm(list(video_dir.glob("*.mp4"))):
        # 从文件名或 sidecar 文件读取 prompt
        prompt = "a cat playing with a ball"  # 替换为实际 prompt
        
        result = evaluator.evaluate(video_file, prompt)
        if result:
            results[video_file.name] = result
    
    # 保存结果
    import json
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)