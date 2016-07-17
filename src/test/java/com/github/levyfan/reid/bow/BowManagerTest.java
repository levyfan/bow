package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.sp.Slic;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLNumericArray;
import com.jmatio.types.MLStructure;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * @author fanliwen
 */
public class BowManagerTest {

    @Test
    public void test() throws IOException, URISyntaxException {
        BufferedImage image = ImageIO.read(this.getClass().getResource("/001_45.bmp"));
        BufferedImage mask = ImageIO.read(this.getClass().getResource("/mask_001_45.bmp"));

        File file = new File(this.getClass().getResource("/sp_codebook.mat").toURI());
        MatFileReader reader = new MatFileReader(file);

        Map<Feature.Type, List<double[]>> codebooks = new EnumMap<>(Feature.Type.class);
        codebooks.put(Feature.Type.HOG, codebook(reader, "codebook_HOG"));
        codebooks.put(Feature.Type.HSV, codebook(reader, "codebook_HSV"));
        codebooks.put(Feature.Type.CN, codebook(reader, "codebook_CN"));
        codebooks.put(Feature.Type.SILTP, codebook(reader, "codebook_SILTP"));

        BowManager bowManager = new BowManager(
                new Bow(10, 3),
                new FeatureManager(),
                new Slic(500, 10),
                8, 8);
        double[] hist = bowManager.fusion(image, mask, codebooks);

        System.out.println(hist.length);
    }

    private List<double[]> codebook(MatFileReader reader, String name) {
        MLStructure mlStructure = (MLStructure) reader.getMLArray(name);
        MLNumericArray ml = (MLNumericArray) mlStructure.getField("wordscenter");

        List<double[]> codebook = new ArrayList<>();
        for (int i = 0; i < ml.getM(); i++) {
            double[] word = new double[ml.getN()];
            for (int j = 0; j < ml.getN(); j++) {
                word[j] = ml.get(i, j).doubleValue();
            }
            codebook.add(word);
        }
        return codebook;
    }
}
