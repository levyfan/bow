package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Bow;
import com.github.levyfan.reid.bow.BowManager;
import com.github.levyfan.reid.bow.ParsingMethod;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.codebook.CodeBook;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.ml.MahalanobisDistance;
import com.github.levyfan.reid.sp.PatchMethod;
import com.github.levyfan.reid.sp.SlicMethod;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Lists;
import com.google.common.io.PatternFilenameFilter;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URISyntaxException;
import java.util.*;
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

    static final boolean wordLevel = false;
    private static final boolean patch = false;

    static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    SuperPixelMethond spMethod;
    FeatureManager featureManager;
    CodeBook codeBook;
    StripMethod stripMethod;
    BowManager bowManager;

    App(File codebookFile) throws IOException, URISyntaxException, ClassNotFoundException {
        Map<Feature.Type, List<double[]>> codebook = Collections.emptyMap();
        Map<Feature.Type, DistanceMeasure> distanceMeasures = new EnumMap<>(Feature.Type.class);
        if (codebookFile != null) {
            if (codebookFile.getName().endsWith("dat")) {
                codebook = this.loadCodeBookDat(codebookFile);
                for (Feature.Type type : Feature.Type.values()) {
                    distanceMeasures.put(type, new EuclideanDistance());
                }
            } else {
                codebook = this.loadCodeBookMat(codebookFile);
                Map<Feature.Type, RealMatrix> mMatrixMap = this.loadKissmeMat(codebookFile);
                for (Feature.Type type : Feature.Type.values()) {
                    if (mMatrixMap.containsKey(type)) {
                        distanceMeasures.put(type, new MahalanobisDistance(mMatrixMap.get(type)));
                    } else {
                        distanceMeasures.put(type, new EuclideanDistance());
                    }
                }
            }
        }

        if (patch) {
            this.spMethod = new PatchMethod(patchSize*4);
        } else {
            this.spMethod = new SlicMethod(numSuperpixels, compactness);
        }

        this.featureManager = new FeatureManager();
        this.codeBook = new CodeBook();
        this.stripMethod = new StripMethod(ystep*4, stripLength*4, pstep*4);
//        this.stripMethod = new ParsingMethod(ystep*4, stripLength*4, pstep*4);
        this.bowManager = new BowManager(
                new Bow(K, sigma, codebook, new Mean(), distanceMeasures),
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
        try {
            MatFileReader reader = new MatFileReader(mat);

            Map<Feature.Type, List<double[]>> codebooks = new EnumMap<>(Feature.Type.class);
            for (Feature.Type type : types) {
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
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private Map<Feature.Type, RealMatrix> loadKissmeMat(File mat) throws IOException {
        try {
            MatFileReader reader = new MatFileReader(mat);

            Map<Feature.Type, RealMatrix> mMatrixMap = new EnumMap<>(Feature.Type.class);
            for (Feature.Type type : types) {
                MLNumericArray ml = (MLNumericArray) reader.getMLArray("M_" + type);
                if (ml != null) {
                    mMatrixMap.put(type, MatrixUtils.from(ml));
                }
            }
            return mMatrixMap;
        } catch (Exception e) {
            e.printStackTrace();
            return Collections.emptyMap();
        }
    }

    List<BowImage> generateHist(File camFolder, File maskFoler, String maskPrefix) throws IOException, ClassNotFoundException {
        File[] camFiles = camFolder.listFiles(filter);
        Arrays.sort(camFiles);
        System.out.println(Arrays.toString(camFiles));

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
