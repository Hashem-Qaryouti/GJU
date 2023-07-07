package Labs;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import imagelib.ImageFrameIJ;
import imagelib.ImageUtil;

public class Lab1 {

    /**
     * Load and show a single grayscale image.
     *
     * @throws Exception
     */
    private void runDemoBasic() throws Exception {
        // load image as grayscale
        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");

        // use global frame in ImageUtil to view image
        // (the frame is created automatically)
        ImageUtil.show(image);

    }



    /**
     * Show two images using frames.
     *
     * @throws Exception
     */
    private void runDemoTwoFrames() throws Exception {
        // load images as grayscale
        // (here the same image is loaded twice)
        ImagePlus image1 = ImageUtil.loadGrayscale("./images/mri.png");
        ImagePlus image2 = ImageUtil.loadGrayscale("./images/mri.png");

        // create first frame and show image
        ImageFrameIJ frame = new ImageFrameIJ("My Viewer 1");
        frame.show(image1);

        // create second frame and show image
        ImageFrameIJ frame2 = new ImageFrameIJ("My Viewer2");
        frame2.show(image2);
    }

    /**
     * Shows single image using the original imageJ functionality.
     *
     * @throws Exception
     */
    private void runDemoImageJ() throws Exception {
        // load image as grayscale
        String currentPath = System.getProperty("user.dir");
        System.out.println(currentPath);
        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");

        // original imageJ show
        image.show();
    }

    /*Calculate the mean of the image intensity*/

    static double getMean(ImagePlus image)
    {
        ImageProcessor ip = image.getProcessor();
        int width = image.getWidth();
        int height = image.getHeight();
        double sum= 0;
        for (int i = 0; i< width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                sum = sum + ip.get(i,j);
            }
        }
        double mean = sum / (width*height);
        return mean;
    }

    static double getVariance(ImagePlus image)
    {
        ImageProcessor ip = image.getProcessor();
        int width = image.getWidth();
        int height = image.getHeight();
        double sum= 0;
        for (int i = 0; i< width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                sum = sum + (ip.get(i,j)*ip.get(i,j));
            }
        }
        double var = sum/(width * height) - getMean(image)*getMean(image);

        return var;
    }

    /**
     * Main method.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // create object
        Lab1 viewer = new Lab1();
        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");

        // run demo
        viewer.runDemoBasic();
        double mean = getMean(image);
        double var = getVariance(image);
        System.out.println("The mean of the image's intensity is "+mean);
        System.out.println("The variance of the image's intensity is "+var);

        //viewer.runDemoTwoFrames();
    }

}
