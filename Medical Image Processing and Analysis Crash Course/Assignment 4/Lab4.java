package Labs;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import imagelib.ImageFrameIJ;
import imagelib.ImageUtil;

public class Lab4{

    /**
     * Load and show a single grayscale image.
     *
     * @throws Exception
     */
    private void runDemoBasic() throws Exception {
        // load image as grayscale
        ImagePlus image1 = ImageUtil.loadGrayscale("./images/channel0.png");
        ImagePlus image2  = ImageUtil.loadGrayscale("./images/channel1.png");
        ImagePlus image3 = ImageUtil.loadGrayscale("./images/channel1.png");


        // use global frame in ImageUtil to view image
        // (the frame is created automatically)
        ImageUtil.show(image1);


    }



    /**
     * Show two images using frames.
     *
     * @throws Exception
     */
    private void runDemoTwoFrames() throws Exception {
        // load images as grayscale
        // (here the same image is loaded twice)
        ImagePlus image1 = ImageUtil.loadGrayscale("./images/channel0.png");
        ImagePlus image2 = ImageUtil.loadGrayscale("./images/channel1.png");
        ImagePlus image3 = ImageUtil.loadGrayscale("./images/channel2.png");
        // create first frame and show image
        ImageFrameIJ frame = new ImageFrameIJ("My Viewer1");
        frame.show(image1);

        // create second frame and show image
        ImageFrameIJ frame2 = new ImageFrameIJ("My Viewer2");
        frame2.show(image2);

        ImageFrameIJ frame3 = new ImageFrameIJ("My Viewer3");
        frame3.show(image3);

    }

    /**
     * Shows single image using the original imageJ functionality.
     *
     * @throws Exception
     */
//    private void runDemoImageJ() throws Exception {
//        // load image as grayscale
//        String currentPath = System.getProperty("user.dir");
//        System.out.println(currentPath);
//        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");
//
//        // original imageJ show
//        image.show();
//    }

    static ImagePlus generateVisualization(ImagePlus image1, ImagePlus image2, ImagePlus image3)
    {
        ImageProcessor ip1 = image1.getProcessor();
        ImageProcessor ip2 = image2.getProcessor();
        ImageProcessor ip3 = image3.getProcessor();

        int rows = ip1.getWidth();
        int cols = ip1.getHeight();

        ImagePlus target = IJ.createImage("Color Image","RGB Black",512,512,1);


        return image1;

    }



    /**
     * Main method.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // create object
        Lab4 viewer = new Lab4();
        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");

        // run demo
        //viewer.runDemoBasic();

        ImagePlus image1 = ImageUtil.loadGrayscale("./images/channel0.png");
        ImagePlus image2 = ImageUtil.loadGrayscale("./images/channel1.png");
        ImagePlus image3 = ImageUtil.loadGrayscale("./images/channel2.png");
        ImageProcessor ip1 = image1.getProcessor();
        ImageProcessor ip2 = image2.getProcessor();
        ImageProcessor ip3 = image3.getProcessor();

        int rows = ip3.getWidth();
        int cols = ip3.getHeight();
        System.out.println("The number of rows is "+rows);
        System.out.println("The number of cols is "+cols);


        viewer.runDemoTwoFrames();
    }

}
