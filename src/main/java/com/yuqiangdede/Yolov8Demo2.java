package com.yuqiangdede;

import ai.onnxruntime.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.List;

/**
 * Demo of yolov8(onnx) prediction using OpenCV 4.9.0 + onnxruntime 1.18.0
 */
public class Yolov8Demo2 {
    public static final String DLL_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\opencv_java490.dll";
    public static final String SO_PATH = "";
    public static final String ONNX_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\yolov8n.onnx";
    public static final String PIC_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\860.jpg";

    public static final float CONF_THRESHOLD = 0.2f;

    static final Model yolomodel;
    private static OrtEnvironment environment;

    static {
        try {
            String osName = System.getProperty("os.name").toLowerCase();
            if (osName.contains("win")) {
                System.load(DLL_PATH);
            } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix")) {
                System.load(SO_PATH);
            } else {
                throw new UnsupportedOperationException("Unsupported operating system: " + osName);
            }
            environment = OrtEnvironment.getEnvironment();

            yolomodel = load(ONNX_PATH);

        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }


    public static void main(String[] args) throws IOException {

        Mat mat = Imgcodecs.imread(PIC_PATH);
        //predictor
        List<ArrayList<Float>> bs = predictor(mat);

        // print
        for (ArrayList<Float> b : bs) {
            System.out.println(b);
        }

        // draw
        BufferedImage image = ImageIO.read(new File(PIC_PATH));
        BufferedImage out = drawImage(image, bs);
        displayImage(out," ");

    }

    /**
     * Predicts bounding boxes and corresponding class labels for the given input image using the YOLO model.
     *
     * @param src The input image in OpenCV Mat format.
     * @return A list of predicted results, containing ArrayList<Float> for each bounding box with coordinates, confidence scores, and class labels.
     */
    private static List<ArrayList<Float>> predictor(Mat src) {

        // pretreatment to OnnxTensor
        try (OnnxTensor tensor = transferTensor(src)) {

            OrtSession.Result result = yolomodel.session.run(Collections.singletonMap("images", tensor));
            OnnxTensor res = (OnnxTensor) result.get(0);

            float[][] data = ((float[][][]) res.getValue())[0];
            // The length of each row is 6, representing the center point, width, height, confidence score, and class label.
            // Fill in the first four values first, and transpose the xywh part into the array.
            Float[][] transpositionData = new Float[data[0].length][6];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    transpositionData[j][i] = data[i][j];
                }
            }
            // Save the highest confidence score and corresponding class index for each bounding box.
            // The fifth element represents the confidence score, and the sixth element represents the class label.
            for (int i = 0; i < data[0].length; i++) {
                for (int j = 4; j < data.length; j++) {
                    // If the set confidence score is smaller than the original data, replace it with the original data, along with the class label.
                    if (transpositionData[i][4] == null || transpositionData[i][4] < data[j][i]) {
                        transpositionData[i][4] = data[j][i]; // confidence score
                        transpositionData[i][5] = (float) (j - 4); // class label
                    }
                }
            }

            List<ArrayList<Float>> boxes = new ArrayList<>();
            // Since the image used for prediction is resized, the coordinates returned are relative to the resized image.
            // Therefore, the final coordinates need to be restored to the original scale.
            float scaleW = (float) src.width() / yolomodel.netWidth;
            float scaleH = (float) src.height() / yolomodel.netHeight;
            // Apply confidence threshold, convert xywh to xyxy, and restore the resized coordinates.
            for (Float[] d : transpositionData) {
                // Apply confidence threshold
                if (d[4] > yolomodel.confThreshold) {
                    // xywh to xyxy
                    d[0] = d[0] - d[2] / 2;
                    d[1] = d[1] - d[3] / 2;
                    d[2] = d[0] + d[2];
                    d[3] = d[1] + d[3];
                    // Restore the resized coordinates to obtain the original coordinates
                    d[0] = d[0] * scaleW;
                    d[1] = d[1] * scaleH;
                    d[2] = d[2] * scaleW;
                    d[3] = d[3] * scaleH;
                    ArrayList<Float> box = new ArrayList<>(Arrays.asList(d));
                    boxes.add(box);
                }
            }

            return NMS(yolomodel, boxes);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Converts an OpenCV Mat object to an OnnxTensor object for model input.
     *
     * @param src The source Mat object containing the image data to be converted.
     * @return The converted OnnxTensor object containing the resized and color space adjusted image data.
     * @throws OrtException Throws when an error occurs in the ORT runtime.
     */
    static OnnxTensor transferTensor(Mat src) throws OrtException {
        Mat dst = new Mat();
        // Resize the image to the model's dimensions
        Imgproc.resize(src, dst, new Size(yolomodel.netWidth, yolomodel.netHeight));
        // Convert the image color space from BGR to RGB, as the model was trained with RGB images
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
        // Convert the image data type to 32-bit floating point and normalize each pixel value to be between 0 and 1
        dst.convertTo(dst, CvType.CV_32FC3, 1. / 255);
        // Create an array to store the adjusted image data in width*height*channels format
        float[] whc = new float[Long.valueOf(yolomodel.channels).intValue() * Long.valueOf(yolomodel.netWidth).intValue() * Long.valueOf(yolomodel.netHeight).intValue()];
        // Get the image data from the Mat object and store it in the array
        dst.get(0, 0, whc);
        // Convert the width-height-channels (WHC) format data to channels-width-height (CHW) format, as the model input requires this format
        float[] chw = whc2cwh(whc);

        // Create an OnnxTensor object using the adjusted image data, which will be used as input for the model
        return OnnxTensor.createTensor(yolomodel.env, FloatBuffer.wrap(chw), new long[]{yolomodel.count, yolomodel.channels, yolomodel.netWidth, yolomodel.netHeight});
    }

    /**
     * Performs Non-Maximum Suppression (NMS) to filter overlapping bounding boxes and keep the ones with the highest confidence scores.
     *
     * @param model The model object containing the NMS threshold
     * @param boxes The list of bounding boxes, where each box is represented by an ArrayList of 6 floats: [x1, y1, x2, y2, score, classIndex]
     * @return The filtered list of bounding boxes after NMS
     */
    static List<ArrayList<Float>> NMS(Model model, List<ArrayList<Float>> boxes) {
        int[] indexs = new int[boxes.size()];
        Arrays.fill(indexs, 1); // Initialize the indexs array with all elements set to 1, indicating all boxes are initially kept

        // Iterate through each box
        for (int cur = 0; cur < boxes.size(); cur++) {
            // Skip if the current box is marked for removal
            if (indexs[cur] == 0) {
                continue;
            }
            ArrayList<Float> curMaxConf = boxes.get(cur); // The current box represents the box with the highest confidence for its class

            // Iterate through the boxes after the current one
            for (int i = cur + 1; i < boxes.size(); i++) {
                // Skip if the current box (being compared) is marked for removal
                if (indexs[i] == 0) {
                    continue;
                }
                float classIndex = boxes.get(i).get(5);

                // Perform NMS if both boxes belong to the same class
                if (classIndex == curMaxConf.get(5)) {
                    // Get the coordinates of both boxes
                    float x1 = curMaxConf.get(0);
                    float y1 = curMaxConf.get(1);
                    float x2 = curMaxConf.get(2);
                    float y2 = curMaxConf.get(3);
                    float x3 = boxes.get(i).get(0);
                    float y3 = boxes.get(i).get(1);
                    float x4 = boxes.get(i).get(2);
                    float y4 = boxes.get(i).get(3);

                    // Skip if the boxes do not overlap
                    if (x1 > x4 || x2 < x3 || y1 > y4 || y2 < y3) {
                        continue;
                    }

                    // Calculate the intersection area between the boxes
                    float intersectionWidth = Math.max(x1, x3) - Math.min(x2, x4);
                    float intersectionHeight = Math.max(y1, y3) - Math.min(y2, y4);
                    float intersectionArea = Math.max(0, intersectionWidth * intersectionHeight);

                    // Calculate the union area of the boxes
                    float unionArea = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersectionArea;

                    // Calculate the Intersection over Union (IoU)
                    float iou = intersectionArea / unionArea;

                    // Mark the box for removal if its IoU is above the threshold
                    indexs[i] = iou > model.nmsThreshold ? 0 : 1;
                }
            }
        }

        // Collect the boxes marked for keeping
        List<ArrayList<Float>> resBoxes = new LinkedList<>();
        for (int index = 0; index < indexs.length; index++) {
            if (indexs[index] == 1) {
                resBoxes.add(boxes.get(index));
            }
        }

        return resBoxes;
    }

    /**
     * Loads an ONNX model from the specified path and returns a Model object containing the model information.
     *
     * @param path The path to the ONNX model.
     * @return The loaded Model object.
     * @throws OrtException Throws an OrtException if an error occurs while loading the model.
     */
    private static Model load(String path) throws OrtException {

        OrtSession session = environment.createSession(path, new OrtSession.SessionOptions());
        Map<String, NodeInfo> infoMap = session.getInputInfo();
        TensorInfo nodeInfo = (TensorInfo) infoMap.get("images").getInfo();

        long count = 1;
        long channels = nodeInfo.getShape()[1];
        long netHeight = 640;
        long netWidth = 640;
        float nmsThreshold = 0.5f;

        return new Model(environment, session, count, channels, netHeight, netWidth, CONF_THRESHOLD, nmsThreshold);
    }

    private static float[] whc2cwh(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    /**
     * Draws a series of bounding boxes on the given BufferedImage and labels each box with a class and confidence score.
     *
     * @param image The BufferedImage on which the bounding boxes will be drawn.
     * @param boxs A list of ArrayLists containing the position and class confidence information for each bounding box.
     *             Each ArrayList contains five Float elements representing the top-left x coordinate, y coordinate,
     *             bottom-right x coordinate, y coordinate, and class confidence score of the bounding box.
     * @return The BufferedImage with the drawn bounding boxes and labels.
     */
    public static BufferedImage drawImage(BufferedImage image, List<ArrayList<Float>> boxs) {
        Graphics graphics = image.getGraphics();

        graphics.setFont(new Font("Arial", Font.BOLD, 24));
        for (ArrayList<Float> b : boxs) {
            graphics.setColor(Color.RED);
            graphics.drawRect( b.get(0).intValue(), b.get(1).intValue(), (int) (b.get(2) - b.get(0)), (int) (b.get(3) - b.get(1)));

            graphics.setColor(Color.BLUE);
            graphics.drawString(b.get(5).intValue() +  " : " + b.get(4),  b.get(0).intValue() + 5,  b.get(3).intValue() - 5);
        }
        graphics.dispose();
        return image;
    }

    public static void displayImage(Image img, String title) {
        ImageIcon icon = new ImageIcon(img);
        JFrame frame = new JFrame(title);
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth(null) + 50, img.getHeight(null) + 50);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    static class Model {
        public OrtEnvironment env;
        public OrtSession session;
        public long count;
        public long channels;
        public long netHeight;
        public long netWidth;
        public float confThreshold;
        public float nmsThreshold;

        public Model(OrtEnvironment env, OrtSession session, long count, long channels, long netHeight, long netWidth, float confThreshold, float nmsThreshold) {
            this.env = env;
            this.session = session;
            this.count = count;
            this.channels = channels;
            this.netHeight = netHeight;
            this.netWidth = netWidth;
            this.confThreshold = confThreshold;
            this.nmsThreshold = nmsThreshold;
        }
    }
}
