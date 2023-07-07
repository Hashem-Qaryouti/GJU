package Labs;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import imagelib.ImageFrameIJ;
import imagelib.ImageUtil;

public class Lab2 {

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
        //ImageUtil.show(image);

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
    // Exercise 2
    static ImagePlus IncreaseBrightness(ImagePlus image)
    {
        ImageProcessor ip = image.getProcessor();
        int rows = ip.getHeight();
        int cols = ip.getWidth();
        for (int i = 0 ; i < rows; i++)
        {
            for (int j = 0 ; j < cols; j++)
            {
                if (ip.get(i,j) + (ip.get(i,j) * 50 / 100) > 255)
            {
                ip.set(i,j,255);
            }
                else {
                    ip.set(i, j, ip.get(i, j) + (ip.get(i, j) * 50 / 100));
                }
            }
        }
        return image;
    }

    //Exercise 3
    static int[] ComputeHistogram(ImagePlus image) {
        int[] array = new int[256]; //Initialize an array of the histogram
        ImageProcessor ip = image.getProcessor();
        int rows = ip.getHeight();
        int cols = ip.getWidth();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[ip.get(i,j)] = array[ip.get(i,j)] + 1;
            }
        }
        return array;

    }




    /**
     * Main method.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // create object
        Lab2 viewer = new Lab2();
        ImagePlus image = ImageUtil.loadGrayscale("./images/mri.png");

        // run demo
        viewer.runDemoBasic();
         int[] histo = ComputeHistogram(image);
         for (int counter = 0; counter < 256; counter++)
         {
             System.out.print(" " +histo[counter]);
         }

         //Call the second Function

        ImageUtil.show(image);
        ImagePlus after = new ImagePlus();
         after = IncreaseBrightness(image);

        ImageUtil.show(image);


        viewer.runDemoTwoFrames();
    }

}
