package Labs;

import ij.ImagePlus;
import ij.process.ImageProcessor;
import imagelib.ImageFrameIJ;
import imagelib.ImageUtil;

public class Lab5 {

    /**
     * Load and show a single grayscale image.
     *
     * @throws Exception
     */
    private void runDemoBasic() throws Exception {
        // load image as grayscale
        ImagePlus image = ImageUtil.loadGrayscale("./images/fluo_shading.png");

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

    static ImagePlus BackgroundCorrection(ImagePlus image)
    {


        ImageProcessor ip = image.getProcessor();
        int rows = ip.getHeight();
        int cols = ip.getWidth();

        int min1 = ip.get(0,0);
        int[] pos1 = new int[2];
        //Get the position of the smallest intensity in the first quadratic image
        for (int i = 0; i < rows/2 ;i++)
        {
            for (int j = 0; j <cols/2;j++)
            {
                if (ip.get(i,j) < min1)
                {
                    min1 = ip.get(i,j);
                    pos1[0] = i;
                    pos1[1] = j;
                }
            }
        }
        //The equation of the first
         System.out.println("**********First***********");
        System.out.println("The value of the intensity is "+ min1+" and the position is "+pos1[0]+" "+pos1[1]);

        //Get the position of the smallest intensity in the third quadratic image
        int min2 = ip.get(rows/2,0);
        int[] pos2 = {rows/2,0};

        for (int i = rows/2; i < rows ;i++)
        {
            for (int j = 0; j <cols/2;j++)
            {
                if (ip.get(i,j) < min2)
                {
                    min2 = ip.get(i,j);
                    pos2[0] = i;
                    pos2[1] = j;
                }
            }
        }

        System.out.println("**********Third***********");
        System.out.println("The value of the intensity is "+ min2+" and the position is "+pos2[0]+" "+pos2[1]);

        //Get the position of the smallest intensity in the second quadratic image
        int min3 = ip.get(0,cols/2);
        int[] pos3 = {0,cols/2};

        for (int i = 0; i < rows/2 ;i++)
        {
            for (int j = cols/2; j < cols;j++)
            {
                if (ip.get(i,j) < min3)
                {
                    min3 = ip.get(i,j);
                    pos3[0] = i;
                    pos3[1] = j;
                }
            }
        }

        System.out.println("**********Second***********");
        System.out.println("The value of the intensity is "+ min3+" and the position is "+pos3[0]+" "+pos3[1]);

        //Get the position of the smallest intensity in the fourth quadratic image
        int min4 = ip.get(rows/2,cols/2);
        int[] pos4 = {rows/2,cols/2};

        for (int i = rows/2; i < rows ;i++)
        {
            for (int j = cols/2; j < cols;j++)
            {
                if (ip.get(i,j) < min4)
                {
                    min4 = ip.get(i,j);
                    pos4[0] = i;
                    pos4[1] = j;
                }
            }
        }


        System.out.println("**********Fourth***********");
        System.out.println("The value of the intensity is "+ min4+" and the position is "+pos4[0]+" "+pos4[1]);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols ;j++)
            {
                int diff = ip.get(i,j) - (int)(10 + 0.0976*i+0.0976*j);
                ip.set(i,j,diff);

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
        Lab5 viewer = new Lab5();
        ImagePlus image = ImageUtil.loadGrayscale("./images/fluo_shading.png");

        // run demo
        viewer.runDemoBasic();
        ImagePlus new_image = BackgroundCorrection(image);
        ImageUtil.show(new_image);

        //viewer.runDemoTwoFrames();
    }

}
