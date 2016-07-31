package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Bow;
import com.github.levyfan.reid.bow.BowManager;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.codebook.CodeBook;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.sp.PatchMethod;
import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.google.common.collect.Lists;
import com.google.common.io.PatternFilenameFilter;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Hello world!
 *
 */
public class App {
    public static final Feature.Type[] types = new Feature.Type[]{
            Feature.Type.HSV, Feature.Type.CN, Feature.Type.HOG, Feature.Type.SILTP};

    static final int numSuperpixels = 500;
    static final double compactness = 20;
    static final int codeBookSize = 350;

    private static final int K = 10;
    private static final double sigma = 3;
    private static final int ystep = 8;
    private static final int stripLength = 8;
    private static final int patchSize = 4;
    static final int pstep = 0;

    private static final File codebookFile = new File("codebook_wordlevel_slic_500_20.0.dat");

    static final boolean wordLevel = false;
    private static final boolean patch = false;

    static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    SuperPixelMethond spMethod;
    FeatureManager featureManager;
    CodeBook codeBook;
    private StripMethod stripMethod;
    private BowManager bowManager;

    App() throws IOException, URISyntaxException, ClassNotFoundException {
        Map<Feature.Type, List<double[]>> codebook;
        if (codebookFile.getName().endsWith("dat")) {
            codebook = this.loadCodeBookDat(codebookFile);
        } else {
            codebook = this.loadCodeBookMat(codebookFile);
        }

        if (patch) {
            this.spMethod = new PatchMethod(patchSize*4);
        } else {
            this.spMethod = new Slic(numSuperpixels, compactness);
        }

        this.featureManager = new FeatureManager();
        this.codeBook = new CodeBook();
        this.stripMethod = new StripMethod(ystep*4, stripLength*4, pstep*4);
        this.bowManager = new BowManager(
                new Bow(K, sigma, codebook, new Mean()),
                this.featureManager,
                wordLevel);

        // output parameters
        System.out.println(this);
    }

    @SuppressWarnings("unchecked")
    private Map<Feature.Type, List<double[]>> loadCodeBookDat(File dat) throws ClassNotFoundException {
        try (ObjectInputStream is = new ObjectInputStream(new FileInputStream(dat))) {
            return  (Map<Feature.Type, List<double[]>>) is.readObject();
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private Map<Feature.Type, List<double[]>> loadCodeBookMat(File mat) throws URISyntaxException, IOException {
        MatFileReader reader = new MatFileReader(mat);

        Map<Feature.Type, List<double[]>> codebooks = new EnumMap<>(Feature.Type.class);
        for (Feature.Type type : Feature.Type.values()) {
            MLNumericArray ml = (MLNumericArray) reader.getMLArray("codebook_" + type);

            List<double[]> codebook = new ArrayList<>();
            for (int i = 0; i < ml.getM(); i++) {
                double[] word = new double[ml.getN()];
                for (int j = 0; j < ml.getN(); j++) {
                    word[j] = ml.get(i, j).doubleValue();
                }
                codebook.add(word);
            }
            codebooks.put(type, codebook);
        }
        return codebooks;
    }

    List<BowImage> generateHist(File camFolder, File maskFoler, String maskPrefix) throws IOException, ClassNotFoundException {
        File[] camFiles = camFolder.listFiles(filter);
        return Lists.newArrayList(camFiles)
                .parallelStream()
                .map(camFile -> {
                    try {
                        System.out.println(camFile.getName());

                        BufferedImage image = ImageIO.read(camFile);
                        String maskFileName = maskPrefix + camFile.getName().replace("jpg", "bmp");
                        BufferedImage mask = ImageIO.read(new File(maskFoler, maskFileName));

                        BowImage bowImage = new BowImage(spMethod, stripMethod, image, mask);
                        bowManager.bow(bowImage);
                        return bowImage;
                    } catch (Exception e) {
                        System.err.println("cam file = " + camFile);
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    Iterable<double[]> fusion(List<BowImage> bowImages, Feature.Type[] types) {
        return bowImages.parallelStream().map(bowImage -> {
            double[][] features = new double[types.length][];
            for (int i = 0; i < types.length; i++) {
                features[i] = bowImage.hist.get(types[i]);
            }
            double[] hist = Doubles.concat(features);

            // normalize
            double sum = Math.sqrt(new SumOfSquares().evaluate(hist));
            for (int i = 0; i < hist.length; i++) {
                hist[i] = hist[i] / sum;
            }
            return hist;
        }).collect(Collectors.toList());
    }

    @Override
    public String toString() {
        return "App{" +
                "spMethod=" + spMethod +
                ", featureManager=" + featureManager +
                ", codeBook=" + codeBook +
                ", stripMethod=" + stripMethod +
                ", bowManager=" + bowManager +
                '}';
    }
}
