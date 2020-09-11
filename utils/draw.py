from torchvision.utils import make_grid, save_image

def draw_grid(X, img_path):
    image_grid = make_grid(X, 10)
    save_image(image_grid, img_path)