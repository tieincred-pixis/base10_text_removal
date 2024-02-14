import cv2
import numpy as np


class Image_mask:
    def __init__(self, image, in_image, coords):
        try:
            # Load the input image
            print(f'Loading the image {image}')
            # self.image = cv2.imread(image[0])
            self.image = cv2.imread(image)
            self.in_image = cv2.imread(in_image)
            print(type(self.image))
        except Exception as e:
            print(e)
            self.image = image
        self.vertices = self.convert2poly(coords)
        

    def convert2poly(self, coords):
        n_vertices = []
        print(len(coords))
        for coord in coords:
            print(len(coord))
            coord = list(coord)
            try:
                # Convert the flat list of coordinates to a list of tuples of (x,y) coordinate pairs
                vertices = [(np.round(coord[i].cpu().numpy()), np.round(coord[i+1].cpu().numpy())) for i in range(0, len(coord), 2)]
            except:
                vertices = [(np.round(coord[i]), np.round(coord[i+1])) for i in range(0, len(coord), 2)]
            # Append the first vertex to the end to close the polygon
            n_vertices.append(vertices)
        return n_vertices

    
    def get_mask(self, out_name, train=False):
        polygon_vertices = [np.array(ver, dtype=np.int32) for ver in self.vertices]
        # print(polygon_vertices[1])
        # Create a mask with the same size as the image, and fill it with zeros
        # print(type(self.image))
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # Draw the polygons on the mask with white color (255)
        cv2.fillPoly(mask, polygon_vertices, 255)
        # Apply the mask to the input image using bitwise AND, if provide
        if train:
            out_train = 'gt_masks/'+out_name.split('/')[-1]
            cropped_image_train = cv2.bitwise_and(self.in_image, self.in_image, mask=mask)
            # cv2.imwrite('res_trn.png', result_train)
            cv2.imwrite(out_train, cropped_image_train)
        # Apply the mask to the image using bitwise AND
        cropped_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        cv2.imwrite('real_data/'+out_name.split('/')[-1], cropped_image)
        # Create a black background image with the same size as the input image
        background = np.zeros_like(self.image)
        # Draw the polygons on the background with white color (255)
        cv2.fillPoly(background, polygon_vertices, 255)
        # Apply the mask to the background using bitwise AND
        background = cv2.bitwise_and(background, background, mask=mask)
        # Combine the cropped image and the background image using bitwise OR
        result = cv2.bitwise_or(cropped_image, background)
        # Save the result image
        # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(out_name, result)
        return result
    
from mmocr.utils import poly2bbox
coords = [[215.83897158322057, 276.00624999999997, 215.83897158322057, 256.25624999999997, 281.0351826792963, 256.25624999999997, 281.0351826792963, 276.00624999999997], [215.34506089309878, 252.79999999999998, 215.34506089309878, 233.04999999999998, 283.50473612990527, 233.04999999999998, 283.50473612990527, 252.79999999999998], [214.3572395128552, 230.58124999999998, 214.3572395128552, 208.36249999999998, 247.44925575101487, 208.36249999999998, 247.44925575101487, 230.58124999999998], [105.20297699594046, 143.1875, 105.20297699594046, 133.3125, 196.08254397834912, 133.3125, 196.08254397834912, 143.1875], [202.00947225981054, 144.66875, 202.00947225981054, 131.83124999999998, 231.15020297699593, 131.83124999999998, 231.15020297699593, 144.66875], [156.07576300552637, 126.89375, 156.07576300552637, 116.52499999999999, 210.40593891892283, 116.52499999999999, 210.40593891892283, 126.89375], [103.72124492557509, 128.375, 103.72124492557509, 115.5375, 132.3680649526387, 115.5375, 132.3680649526387, 128.375], [135.8254397834912, 115.04374999999999, 151.13667117726658, 115.04374999999999, 151.13667117726658, 128.375, 135.8254397834912, 128.375], [104.70906630581867, 100.23124999999999, 170.89309878213803, 100.23124999999999, 170.89309878213803, 112.08125, 104.70906630581867, 112.08125], [177.80784844384303, 110.6, 177.80784844384303, 100.23124999999999, 230.65629228687413, 100.23124999999999, 230.65629228687413, 110.6], [235.59539918809202, 113.56249999999999, 235.59539918809202, 98.75, 254.85791610284167, 98.75, 254.85791610284167, 113.56249999999999], [105.69688768606224, 84.43124999999999, 192.13125845737483, 84.43124999999999, 192.13125845737483, 94.8, 105.69688768606224, 94.8], [534.3313594503235, 108.65067596435546, 540.2639551640202, 70.10107841491698, 644.0737084993974, 86.06661605834961, 638.1411127857007, 124.61621360778808], [405.16629808810467, 85.84358625411987, 513.7008208609079, 69.86822423934936, 519.1698929679571, 107.00018911361694, 410.6353701951539, 122.97555112838744], [150.64276048714478, 78.50625, 150.64276048714478, 67.64375, 229.66847090663057, 67.64375, 229.66847090663057, 78.50625], [104.70906630581867, 79.49374999999999, 104.70906630581867, 66.65625, 143.72801082543978, 66.65625, 143.72801082543978, 79.49374999999999], [135.8254397834912, 61.224999999999994, 135.8254397834912, 50.3625, 191.14343707713124, 50.3625, 191.14343707713124, 61.224999999999994], [103.72124492557509, 49.375, 130.39242219215154, 49.375, 130.39242219215154, 62.2125, 103.72124492557509, 62.2125], [528.4517602572099, 58.884467840194695, 531.7550291647285, 33.498525857925415, 615.140080226129, 44.34183669090271, 611.8368113186104, 69.72778244018554], [427.5232330033195, 45.05825109481811, 515.399784403015, 34.84336853027344, 517.9220329150456, 56.52779026031494, 430.0454815153501, 66.74267282485961], [104.21515561569689, 45.424996232986445, 104.21515561569689, 33.081246232986445, 142.7401894451962, 33.081246232986445, 142.7401894451962, 45.424996232986445], [325.4871447902571, 6.418749999999999, 375.3721244925575, 6.418749999999999, 375.3721244925575, 22.7125, 325.4871447902571, 22.7125]]
# coords = [poly2bbox(poly) for poly in polys]
mask_getter = Image_mask('1.jpg','1.jpg',coords)

mask_getter.get_mask('maske.png')