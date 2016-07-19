package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Bow;
import com.github.levyfan.reid.bow.BowManager;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.codebook.CodeBook;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.pca.BlasPca;
import com.github.levyfan.reid.sp.PatchMethod;
import com.github.levyfan.reid.sp.Slic;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.github.levyfan.reid.viper.Viper;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.PatternFilenameFilter;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import com.jmatio.types.MLStructure;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Hello world!
 *
 */
public class App 
{
    static final Feature.Type[] types = new Feature.Type[]{
            Feature.Type.HSV, Feature.Type.CN, Feature.Type.HOG, Feature.Type.SILTP};

    static final File training = new File("/data/reid/TUDpositive");
    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    static final int numSuperpixels = 500;
    static final double compactness = 20;
    private static final int codeBookSize = 350*4;

    private static final int K = 10;
    private static final double sigma = 3;
    private static final int ystep = 8;
    private static final int stripLength = 8;
    private static final int patchSize = 4;

    static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    SuperPixelMethond spMethod;
    FeatureManager featureManager;
    CodeBook codeBook;
    private StripMethod stripMethod;
    private BowManager bowManager;

    App() throws IOException, URISyntaxException, ClassNotFoundException {
        Map<Feature.Type, List<double[]>> codebook = this.loadCodeBookDat(
                new File("codebook_slic_300_20.0.dat"));
//        Map<Feature.Type, List<double[]>> codebook = app.loadCodeBookMat();

        this.spMethod = new Slic(numSuperpixels, compactness);
//        this.spMethod = new PatchMethod(patchSize*4);

        this.featureManager = new FeatureManager();
        this.codeBook = new CodeBook(codeBookSize);
        this.stripMethod = new StripMethod(ystep*4, stripLength*4);
        this.bowManager = new BowManager(new Bow(K, sigma, codebook), this.featureManager);
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

    private Map<Feature.Type, List<double[]>> loadCodeBookMat() throws URISyntaxException, IOException {
        File file = new File(this.getClass().getResource("/sp_codebook.mat").toURI());
        MatFileReader reader = new MatFileReader(file);

        Map<Feature.Type, List<double[]>> codebooks = new EnumMap<>(Feature.Type.class);
        //FIXME
        for (Feature.Type type : Feature.Type.values()) {
            MLStructure mlStructure = (MLStructure) reader.getMLArray("codebook_" + type);
            MLNumericArray ml = (MLNumericArray) mlStructure.getField("wordscenter");

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

    private List<BowImage> generateHist(File camFolder, File maskFoler) throws IOException, ClassNotFoundException {
        File[] camFiles = camFolder.listFiles(filter);
        return Lists.newArrayList(camFiles)
                .parallelStream()
                .map(camFile -> {
                    try {
                        BufferedImage image = ImageIO.read(camFile);
                        BufferedImage mask = ImageIO.read(new File(maskFoler, "mask_" + camFile.getName()));
                        System.out.println(camFile.getName());

                        BowImage bowImage = new BowImage(spMethod, stripMethod, image, mask);
                        bowManager.bow(bowImage);
                        return bowImage;
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    private List<double[]> fusion(List<BowImage> bowImages, Feature.Type[] types) {
        return bowImages.stream().map(bowImage -> {
            double[][] features = new double[types.length][];
            for (int i = 0; i < types.length; i++) {
                features[i] = bowImage.hist.get(types[i]);
            }
            return Doubles.concat(features);
        }).collect(Collectors.toList());
    }

    private RealMatrix calculateScore(List<double[]> histA, List<double[]> histB, boolean enablePCA) {
        histA.stream().forEach(hist -> {
            double sum = Math.sqrt(new SumOfSquares().evaluate(hist));
            for (int i = 0; i < hist.length; i++) {
                hist[i] = hist[i] / sum;
            }
        });
        histB.stream().forEach(hist -> {
            double sum = Math.sqrt(new SumOfSquares().evaluate(hist));
            for (int i = 0; i < hist.length; i++) {
                hist[i] = hist[i] / sum;
            }
        });

        if (enablePCA) {
            List<double[]> hist = Lists.newArrayList(Iterables.concat(histA, histB));
            BlasPca pca = new BlasPca(hist);

            DoubleMatrix histPcaA = pca.ux.get(
                    new IntervalRange(0, pca.ux.getRows()), new IntervalRange(0, histA.size()));
            DoubleMatrix histPcaB = pca.ux.get(
                    new IntervalRange(0, pca.ux.getRows()), new IntervalRange(histA.size(), hist.size()));
            return MatrixUtils.createRealMatrix(
                    histPcaA.transpose().mmul(histPcaB).toArray2());
        } else {
            return MatrixUtils.createRealMatrix(histA.toArray(new double[0][]))
                    .multiply(MatrixUtils.createRealMatrix(histB.toArray(new double[0][])).transpose());
        }
    }

    public static void main( String[] args ) throws IOException, URISyntaxException, ClassNotFoundException {
        App app = new App();

        List<BowImage> bowImagesA = app.generateHist(testingA, maskA);
        List<BowImage> bowImagesB = app.generateHist(testingB, maskB);

        List<double[]> histA = app.fusion(bowImagesA, types);
        List<double[]> histB = app.fusion(bowImagesB, types);

        MLDouble a = new MLDouble("HistA", histA.toArray(new double[0][]));
        MLDouble b = new MLDouble("HistB", histB.toArray(new double[0][]));
        new MatFileWriter().write(
                "hist_" + numSuperpixels + "_" + compactness + ".mat",
                Arrays.asList(a, b));

        // non-pca
        RealMatrix score = app.calculateScore(histA, histB, false);
        double[] MR = new Viper().eval(score);
        System.out.println("fusion not_pca:" + Doubles.asList(MR).subList(0,50));

        // pca
        score = app.calculateScore(histA, histB, true);
        MR = new Viper().eval(score);
        System.out.println("fusion pca:" + Doubles.asList(MR).subList(0,50));

        for (Feature.Type type : Feature.Type.values()) {
            List<double[]> histTypeA = app.fusion(bowImagesA, new Feature.Type[]{type});
            List<double[]> histTypeB = app.fusion(bowImagesB, new Feature.Type[]{type});

            RealMatrix scoreType = app.calculateScore(histTypeA, histTypeB, false);
            double[] scoreMR = new Viper().eval(scoreType);
            System.out.println(type + ":" + Doubles.asList(scoreMR).subList(0,50));
        }
    }
}
