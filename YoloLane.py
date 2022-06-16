import cv2
import numpy as np

# we are not going to bother with objects less than 30% probability
THRESHOLD = 0.5
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.4
YOLO_IMAGE_SIZE = 320


# Lane Detection Function 1

def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # we have to turn the image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)

    # we are interested in the "lower region" of the image (there are the driving lanes)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height * 0.65),
        (width, height)
    ]

    # we can get rid of the un-relevant part of the image
    # we just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # use the line detection algorithm (radians instead of degrees 1 degree = pi / 180)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=40, maxLineGap=150)

    # draw the lines on the image
    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines


def draw_the_lines(image, lines):
    # create a distinct image for the lines [0,255] - all 0 values means black image
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are (x,y) for the starting and end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    # finally we have to merge the image with the lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)

    return image_with_lines


def region_of_interest(image, region_points):
    # we are going to replace pixels with 0 (black) - the regions we are not interested
    mask = np.zeros_like(image)
    # the region that we are interested in is the lower triangle - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)
    # we have to use the mask: we want to keep the regions of the original image where
    # the mask has white colored pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def find_objects(model_outputs):
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # the center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


def show_detected_objects(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio,
                          height_ratio):
    for index in bounding_box_ids:
        print('index', index)
        bounding_box = all_bounding_boxes[index]
        print(bounding_box)
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        # we have to transform the locations adn coordinates because the resized image
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        # OpenCV deals with BGR blue green red (255,0,0) then it is the blue color
        # we are not going to detect every objects just PERSON and CAR
        if class_ids[index] == 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_with_confidence = 'CAR' + str(int(confidence_values[index] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

        if class_ids[index] == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_with_confidence = 'PERSON' + str(int(confidence_values[index] * 100)) + '%'
            cv2.putText(img, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)


capture = cv2.VideoCapture('expressway.mp4')

# there are 80 (90) possible output classes
# 0: person - 2: car - 5: bus
classes = ['car', 'person', 'bus']

neural_network = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
# define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:

    # after reading a frame (so basically one image) we just have to do
    # the same operations as with single images !!!
    frame_grabbed, frame = capture.read()

    if not frame_grabbed:
        break

    # Lane detection function
    frame = get_detected_lanes(frame)

    original_width, original_height = frame.shape[1], frame.shape[0]

    # the image into a BLOB [0-1] RGB - BGR
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
    neural_network.setInput(blob)

    layer_names = neural_network.getLayerNames()
    # YOLO network has 3 output layer - note: these indexes are starting with 1
    # output_names = [layer_names[index[0] - 1] for index in neural_network.getUnconnectedOutLayers()]
    output_names = [layer_names[index - 1] for index in neural_network.getUnconnectedOutLayers()]

    outputs = neural_network.forward(output_names)
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)
    show_detected_objects(frame, predicted_objects, bbox_locations, class_label_ids, conf_values,
                          original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)

    cv2.imshow('YOLO Algorithm', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
