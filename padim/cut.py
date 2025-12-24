import cv2
import os

# 全局变量
center_x, center_y = 500, 500  # 初始中心点坐标
crop_width, crop_height = 50, 50  # 裁剪区域宽高
drawing = False  # 标记鼠标是否在拖动


def crop_by_center(img, center_x, center_y, width, height):
    """
    根据中心点裁剪图像
    :param img: 输入图像
    :param center_x: 裁剪区域中心点的 x 坐标
    :param center_y: 裁剪区域中心点的 y 坐标
    :param width: 裁剪区域的宽度
    :param height: 裁剪区域的高度
    :return: 裁剪后的图像
    """
    h_img, w_img = img.shape[:2]

    # 计算裁剪区域的左上角和右下角坐标
    x1 = max(0, center_x - width // 2)
    y1 = max(0, center_y - height // 2)
    x2 = min(w_img, center_x + width // 2)
    y2 = min(h_img, center_y + height // 2)

    # 裁剪图像
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def draw_rectangle(event, x, y, flags, param):
    """
    鼠标回调函数，用于更新裁剪框的位置
    """
    global center_x, center_y, drawing

    if event == cv2.EVENT_MOUSEMOVE and drawing:
        center_x, center_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    if event == cv2.EVENT_LBUTTONUP:
        drawing = False


if __name__ == "__main__":
    # 示例：根据中心点裁剪
    image_path = "E:\\Code\\Padim\\dataset\\jinyuan4\\001.bmp"  # 替换为你的图像路径
    img = cv2.imread(image_path)

    if img is None:
        print("无法加载图像，请检查路径！")
    else:
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", draw_rectangle)

        while True:
            # 显示图像并绘制裁剪框
            temp_img = img.copy()
            x1 = max(0, center_x - crop_width // 2)
            y1 = max(0, center_y - crop_height // 2)
            x2 = min(temp_img.shape[1], center_x + crop_width // 2)
            y2 = min(temp_img.shape[0], center_y + crop_height // 2)
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Image", temp_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # 按下 Enter 键确认裁剪
                cropped = crop_by_center(img, center_x, center_y, crop_width, crop_height)
                output_path = "E:\\Code\\Padim\\dataset\\jinyuan4\\cut.bmp"
                cv2.imwrite(output_path, cropped)
                print(f"裁剪结果已保存到: {output_path}")
                break
            elif key == 27:  # 按下 Esc 键退出
                print("操作已取消。")
                break

        cv2.destroyAllWindows()