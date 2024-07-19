package com.yuqiangdede;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.List;

/**
 * Demo of yolov8(onnx) prediction using OpenCV 4.9.0
 * <p>
 * https://blog.csdn.net/taoli188/article/details/134720614
 */
public class Yolov8Demo1 {
    public static final String DLL_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\opencv_java490.dll";
    public static final String SO_PATH = "";
    public static final String ONNX_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\yolov8n.onnx";
    public static final String PIC_PATH = "E:\\JavaCode\\java-yolo-onnx\\src\\main\\resources\\860.jpg";

    static {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("win")) {
            System.load(DLL_PATH);
        } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix")) {
            System.load(SO_PATH);
        } else {
            throw new UnsupportedOperationException("Unsupported operating system: " + osName);
        }
    }

    public static void main(String[] args) {
        // Read the ONNX model from the given file path
        Net net = Dnn.readNetFromONNX(ONNX_PATH);

        // Read the image from the file path
        Mat mat = Imgcodecs.imread(PIC_PATH);

        // Convert the image to a blob for network input
        Mat blob = Dnn.blobFromImage(mat, 1 / 255.0, new Size(640, 640));

        // Set the input for the network
        net.setInput(blob);

        // Perform forward pass through the network
        Mat predict = net.forward();

        // Reshape the prediction output
        Mat mask = predict.reshape(1, predict.size(1));

        // Calculate the scale factors for width and height
        double width = mat.cols() / 640.0;
        double height = mat.rows() / 640.0;

        // Initialize arrays to store the detection results
        Rect2d[] rect2d = new Rect2d[mask.cols()];
        float[] scoref = new float[mask.cols()];
        int[] classid = new int[mask.cols()];

        // Iterate over the detection results
        for (int i = 0; i < mask.cols(); i++) {
            // Extract the bounding box coordinates
            double[] x = mask.col(i).get(0, 0);
            double[] y = mask.col(i).get(1, 0);
            double[] w = mask.col(i).get(2, 0);
            double[] h = mask.col(i).get(3, 0);

            // Calculate the bounding box position in the original image
            rect2d[i] = new Rect2d((x[0] - w[0] / 2) * width, (y[0] - h[0] / 2) * height, w[0] * width, h[0] * height);

            // Extract the score matrix
            Mat score = mask.col(i).submat(4, predict.size(1) - 1, 0, 1);

            // Find the maximum score and its location
            Core.MinMaxLocResult mmr = Core.minMaxLoc(score);

            // Store the score and class ID
            scoref[i] = (float) mmr.maxVal;
            classid[i] = (int) mmr.maxLoc.y;
        }

        // Convert the bounding boxes to OpenCV's Mat format
        MatOfRect2d boxes = new MatOfRect2d(rect2d);
        MatOfFloat scores = new MatOfFloat(scoref);
        MatOfInt indices = new MatOfInt();

        // Perform non-maximum suppression
        Dnn.NMSBoxes(boxes, scores, 0.15f, 0.5f, indices);

        // Check if any detections were found
        if (indices.empty()) {
            System.out.println("No boxes returned from NMS.");
        } else {
            // Iterate over the filtered detections
            List<Integer> result = indices.toList();
            for (Integer integer : result) {
                // Draw the bounding box and class information on the image
                Imgproc.rectangle(mat, new Rect(rect2d[integer].tl(), rect2d[integer].size()),
                        new Scalar(255, 0, 0), 2);
                Imgproc.putText(mat, classid[integer] + " : " + scoref[integer], rect2d[integer].tl(),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0));
            }
        }

        // Display the image
        displayImage(matToBufferedImage(mat), "pic");
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

    public static BufferedImage matToBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] b = new byte[bufferSize];
        matrix.get(0, 0, b);
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }
}
