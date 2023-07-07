package Labs;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import imagelib.ImageFrameIJ;
import imagelib.ImageUtil;

public class Lab3 {

    /**
     * Load and show a single grayscale image.
     *
     * @throws Exception
     */
    private void runDemoBasic() throws Exception {
        // load image as grayscale
        ImagePlus image = ImageUtil.loadGrayscale("./images/level1_herzgewebe_lc.png");

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
        ImagePlus image1 = ImageUtil.loadGrayscale("./images/level1_herzgewebe_lc.png");
        ImagePlus image2 = ImageUtil.loadGrayscale("./images/level1_herzgewebe_lc.png");

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
        ImagePlus image = ImageUtil.loadGrayscale("./images/level1_herzgewebe_lc.png");

        // original imageJ show
        image.show();
    }

    //Exercise 1
    static ImagePlus LinearScaling(ImagePlus image)
    {
        ImageProcessor ip = image.getProcessor();
        int a =ip.get(0,0);
        int b = ip.get(0,0);
        int rows = ip.getHeight();
        int cols = ip.getWidth();
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (ip.get(i,j) < a)
                {
                    a = ip.get(i,j);
                }
            }
        }
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (ip.get(i,j) > b)
                {
                    b = ip.get(i,j);
                }
            }
        }

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0 ; j < rows ;j++)
            {
                if (ip.get(i,j) < a)
                {
                    ip.set(i,j,0);
                }
                else if (ip.get(i,j) >= a && ip.get(i,j) <= b)
                {
                    int value = (255* (ip.get(i,j)-a) / (b-a)) ;
                    ip.set(i,j,value);
                }
                else {
                    ip.set(i,j,255);
                }
            }
        }

        return image;

    }



    /**
     * Main method.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // create object
        Lab3 viewer = new Lab3();
        ImagePlus image = ImageUtil.loadGrayscale("./images/level1_herzgewebe_lc.png");

        ImagePlus new_image = LinearScaling(image);
        ImageUtil.show(new_image);

        // run demo
        //viewer.runDemoBasic();


        //viewer.runDemoTwoFrames();
    }

}
