import cv2
import numpy as np
import os

def non_max_suppression_fast(boxes, overlapThresh):
    """
    非极大值抑制 (NMS)
    """
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def save_matched_regions(image_path, template_path, output_dir, threshold=0.8, overlapThresh=0.3):
    """
    模板匹配并保存经过 NMS 筛选的框
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取图片
    img_rgb = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    if img_rgb is None or template is None:
        print("错误：无法读取图片")
        return

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape[:2]

    print(f"正在匹配... (阈值: {threshold})")
    
    # 模板匹配
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 筛选大于阈值的点
    loc = np.where(res >= threshold)
    
    # 收集所有候选框 [x, y, w, h, score]
    candidates = []
    for pt in zip(*loc[::-1]):
        score = res[pt[1], pt[0]]
        candidates.append([pt[0], pt[1], w, h, score])

    candidates = np.array(candidates)
    print(f"初步候选框数量: {len(candidates)}")

    # 非极大值抑制 (NMS)
    final_boxes = non_max_suppression_fast(candidates, overlapThresh)
    print(f"NMS 筛选后剩余框数量: {len(final_boxes)}")

    # 保存图片
    result_img = img_rgb.copy()
    base_filename = os.path.splitext(os.path.basename(image_path))[0]  # 获取当前文件名作为前缀
    for i, (x, y, w_box, h_box, score) in enumerate(final_boxes):
        # 裁剪
        crop_img = img_rgb[y:y+h_box, x:x+w_box]
        
        # 保存
        save_name = os.path.join(output_dir, f"{base_filename}_{i+1}.jpg")
        cv2.imwrite(save_name, crop_img)
        
        # 画框展示
        cv2.rectangle(result_img, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        # 标号
        cv2.putText(result_img, f"{i+1} ({score:.2f})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 显示结果 (缩放一下方便看)
    h_show, w_show = result_img.shape[:2]
    scale = 800 / max(h_show, w_show) # 缩放到长边 800
    if scale < 1:
        result_img = cv2.resize(result_img, None, fx=scale, fy=scale)
    
    cv2.imshow('Final Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_image_path = r"E:\\Code\\Padim\\dataset\\jinyuan3\\000068.jpg"
    template_image_path = r"E:\\Code\\Padim\\dataset\\jinyuan3\\template.jpg"
    output_folder = r"E:\\Code\\Padim\\dataset\\jinyuan3"
    
    # 阈值建议：0.8 (如果漏检就调低到 0.75)
    save_matched_regions(main_image_path, template_image_path, output_folder, threshold=0.65, overlapThresh=0.3)