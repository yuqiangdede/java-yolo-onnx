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
import java.util.List;
import java.util.*;

/**
 * Demo of yolov8(onnx) prediction using OpenCV 4.9.0 + onnxruntime 1.18.0
 */
public class Yolov8PoseDemo1 {
    public static final String DLL_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\opencv_java490.dll";
    public static final String SO_PATH = "";
    public static final String ONNX_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\yolov8n-pose.onnx";
    public static final String PIC_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\123.jpg";

    public static final float CONF_THRESHOLD = 0.6f;

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
        displayImage(out, " ");

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
            /*
             people      X1    X2    X3    ... Xn
                         Y1    Y2    Y3    ... Yn
                         W1    W2    W3    ... Wn
                         H1    H2    H3    ... Hn
                         C1    C2    C3    ... Cn    -- confidence for people
             point1      X1-1  X2-1  X3-1  ... Xn-1  -- X(people num)-(point num)
                         Y1-1  Y2-1  Y3-1  ... Yn-1
                         C1-1  C2-1  C3-1  ... Cn-1  -- confidence for 1st point
             point2      X1-2  X2-2  X3-2  ... Xn-2
                         Y1-2  Y2-2  Y3-2  ... Yn-2
                         C1-2  C2-2  C3-2  ... Cn-2  -- confidence for 2nd point
             ............................
             point17     X1-17 X2-17 X3-17 ... Xn-17
                         Y1-17 Y2-17 Y3-17 ... Yn-17
                         C1-17 C2-17 C3-17 ... Cn-17 -- confidence for 17th point


             transpositionData
             people1     X1 Y1 W1 H1 C1 X1-1 Y1-1 C1-1 X1-2 Y1-2 C1-2 .......... X1-17 Y1-17 C1-17 -- length is 56
             people2     X2 Y2 W2 H2 C2 X2-1 Y2-1 C2-1 X2-2 Y2-2 C2-2 .......... X2-17 Y2-17 C2-17
             ......................................................................
             peoplem     Xm Ym Wm Hm Cm Xm-1 Ym-1 Cm-1 Xm-2 Ym-2 Cm-2 .......... Xm-17 Ym-17 Cm-17

             */

            Float[][] transpositionData = new Float[data[0].length][56];
            // put X1 Y1 W1 H1 C1
            for (int i = 0; i < 56; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    transpositionData[j][i] = data[i][j];
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
                    d[5] = d[5] * scaleW;
                    d[6] = d[6] * scaleH;
                    d[8] = d[8] * scaleW;
                    d[9] = d[9] * scaleH;
                    d[11] = d[11] * scaleW;
                    d[12] = d[12] * scaleH;
                    d[14] = d[14] * scaleW;
                    d[15] = d[15] * scaleH;
                    d[17] = d[17] * scaleW;
                    d[18] = d[18] * scaleH;
                    d[20] = d[20] * scaleW;
                    d[21] = d[21] * scaleH;
                    d[23] = d[23] * scaleW;
                    d[24] = d[24] * scaleH;
                    d[26] = d[26] * scaleW;
                    d[27] = d[27] * scaleH;
                    d[29] = d[29] * scaleW;
                    d[30] = d[30] * scaleH;
                    d[32] = d[32] * scaleW;
                    d[33] = d[33] * scaleH;
                    d[35] = d[35] * scaleW;
                    d[36] = d[36] * scaleH;
                    d[38] = d[38] * scaleW;
                    d[39] = d[39] * scaleH;
                    d[41] = d[41] * scaleW;
                    d[42] = d[42] * scaleH;
                    d[44] = d[44] * scaleW;
                    d[45] = d[45] * scaleH;
                    d[47] = d[47] * scaleW;
                    d[48] = d[48] * scaleH;
                    d[50] = d[50] * scaleW;
                    d[51] = d[51] * scaleH;
                    d[53] = d[53] * scaleW;
                    d[54] = d[54] * scaleH;
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

        for (int cur = 0; cur < boxes.size(); cur++) {
            // Skip if the current box is marked for removal
            if (indexs[cur] == 0) {
                continue;
            }
            ArrayList<Float> box = boxes.get(cur);

            // Iterate through the boxes after the current one
            for (int i = cur + 1; i < boxes.size(); i++) {
                // Skip if the current box (being compared) is marked for removal
                if (indexs[i] == 0) {
                    continue;
                }

                // Get the coordinates of both boxes
                float x1 = box.get(0);
                float y1 = box.get(1);
                float x2 = box.get(2);
                float y2 = box.get(3);
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
     * @param boxs  A list of ArrayLists containing the position and class confidence information for each bounding box.
     *              Each ArrayList contains five Float elements representing the top-left x coordinate, y coordinate,
     *              bottom-right x coordinate, y coordinate, and class confidence score of the bounding box.
     * @return The BufferedImage with the drawn bounding boxes and labels.
     */
    public static BufferedImage drawImage(BufferedImage image, List<ArrayList<Float>> boxs) {
        Graphics graphics = image.getGraphics();

        graphics.setFont(new Font("Arial", Font.BOLD, 24));
        for (ArrayList<Float> b : boxs) {
            graphics.setColor(Color.RED);
            graphics.drawRect(b.get(0).intValue(), b.get(1).intValue(), (int) (b.get(2) - b.get(0)), (int) (b.get(3) - b.get(1)));

            graphics.setColor(Color.BLUE);
            graphics.drawString(String.valueOf(b.get(4)), b.get(0).intValue() + 5, b.get(3).intValue() - 5);

            graphics.setColor(Color.ORANGE);
            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(8).intValue(), b.get(9).intValue());
            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(11).intValue(), b.get(12).intValue());
            graphics.drawLine(b.get(8).intValue(), b.get(9).intValue(), b.get(14).intValue(), b.get(15).intValue());
            graphics.drawLine(b.get(11).intValue(), b.get(12).intValue(), b.get(17).intValue(), b.get(18).intValue());

            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(20).intValue(), b.get(21).intValue());
            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(23).intValue(), b.get(24).intValue());
            graphics.drawLine(b.get(20).intValue(), b.get(21).intValue(), b.get(26).intValue(), b.get(27).intValue());
            graphics.drawLine(b.get(23).intValue(), b.get(24).intValue(), b.get(29).intValue(), b.get(30).intValue());
            graphics.drawLine(b.get(26).intValue(), b.get(27).intValue(), b.get(32).intValue(), b.get(33).intValue());
            graphics.drawLine(b.get(29).intValue(), b.get(30).intValue(), b.get(35).intValue(), b.get(36).intValue());

            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(38).intValue(), b.get(39).intValue());
            graphics.drawLine(b.get(5).intValue(), b.get(6).intValue(), b.get(41).intValue(), b.get(42).intValue());
            graphics.drawLine(b.get(38).intValue(), b.get(39).intValue(), b.get(44).intValue(), b.get(45).intValue());
            graphics.drawLine(b.get(44).intValue(), b.get(45).intValue(), b.get(50).intValue(), b.get(51).intValue());
            graphics.drawLine(b.get(41).intValue(), b.get(42).intValue(), b.get(47).intValue(), b.get(48).intValue());
            graphics.drawLine(b.get(47).intValue(), b.get(48).intValue(), b.get(53).intValue(), b.get(54).intValue());


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
