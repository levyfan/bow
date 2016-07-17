package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.feature.Hog;
import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixel;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLSingle;
import com.jmatio.types.MLStructure;
import org.apache.commons.math3.util.Pair;
import org.junit.Assert;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class BowTest {

    @Test
    public void testVote() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/bee.jpg"));
        SuperPixel[] sp = new Slic(500, 10).slic(image);

        File file = new File(this.getClass().getResource("/sp_codebook.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLStructure mlStructure = (MLStructure) reader.getMLArray("codebook_HOG");
        MLSingle mlDouble = (MLSingle) mlStructure.getField("wordscenter");

        List<double[]> codebook = new ArrayList<>();
        for (int i = 0; i < mlDouble.getM(); i++) {
            double[] word = new double[mlDouble.getN()];
            for (int j = 0; j < mlDouble.getN(); j++) {
                word[j] = mlDouble.get(i, j);
            }
            codebook.add(word);
        }

        List<double[]> hog = new Hog().hog(image, sp);
        Pair<int[], double[]> pair = Bow.vote(hog.get(0), codebook, 10, 3);
        int[] word = pair.getFirst();
        double[] wword = pair.getSecond();

        Assert.assertEquals(207, word[0]);
        Assert.assertEquals(87, word[1]);
        Assert.assertEquals(273, word[2]);
        Assert.assertEquals(168, word[3]);
        Assert.assertEquals(346, word[4]);

        Assert.assertTrue(Math.abs(wword[0] - 0.9876) < 0.001);
        Assert.assertTrue(Math.abs(wword[1] - 0.9765) < 0.001);
        Assert.assertTrue(Math.abs(wword[2] - 0.9763) < 0.001);
        Assert.assertTrue(Math.abs(wword[3] - 0.9758) < 0.001);
        Assert.assertTrue(Math.abs(wword[4] - 0.9745) < 0.001);
    }

    @Test
    public void test() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/001_45.bmp"));
        BufferedImage mask = ImageIO.read(this.getClass().getResource("/mask_001_45.bmp"));

        File file = new File(this.getClass().getResource("/sp_codebook.mat").toURI());
        MatFileReader reader = new MatFileReader(file);
        MLStructure mlStructure = (MLStructure) reader.getMLArray("codebook_HOG");
        MLSingle mlDouble = (MLSingle) mlStructure.getField("wordscenter");

        List<double[]> codebook = new ArrayList<>();
        for (int i = 0; i < mlDouble.getM(); i++) {
            double[] word = new double[mlDouble.getN()];
            for (int j = 0; j < mlDouble.getN(); j++) {
                word[j] = mlDouble.get(i, j);
            }
            codebook.add(word);
        }

        SuperPixel[] sp = new Slic(500, 10).slic(image);
        Strip[] strip = new StripMethod(8, 8).strip(sp, mask);
        List<double[]> feature = new Hog().extract(image, sp);

        List<double[]> hist = new Bow(10, 3).bow(strip, feature, codebook);

        Assert.assertTrue(Math.abs(hist.get(0)[1] - 0.1758) < 0.01);
        Assert.assertTrue(Math.abs(hist.get(0)[2] - 0.2306) < 0.01);
        Assert.assertTrue(Math.abs(hist.get(0)[11] - 0.3962) < 0.01);
    }
}
