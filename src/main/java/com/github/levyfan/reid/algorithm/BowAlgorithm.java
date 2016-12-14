package com.github.levyfan.reid.algorithm;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.bow.Bow;
import com.github.levyfan.reid.bow.BowManager;
import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.bow.StripMethod;
import com.github.levyfan.reid.codebook.CodeBook;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.github.levyfan.reid.ml.KissMe;
import com.github.levyfan.reid.ml.MahalanobisDistance;
import com.github.levyfan.reid.sp.SlicMethod;
import com.github.levyfan.reid.sp.SuperPixelMethond;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.ParallelKMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class BowAlgorithm implements Algorithm {

    public static final Feature.Type[] types = new Feature.Type[]{
            Feature.Type.HSV, Feature.Type.CN, Feature.Type.HOG, Feature.Type.SILTP};

    private SuperPixelMethond spMethod;
    private StripMethod stripMethod;

    private FeatureManager featureManager;
    private BowManager bowManager;

    private KissMe kissMe = new KissMe();

    private static final int numSuperpixels = 500;
    private static final double compactness = 20;

    private static final int K = 10;
    private static final double sigma = 3;
    private static final int ystep = 8;
    private static final int stripLength = 8;
    private static final int patchSize = 4;
    private static final int pstep = 0;

    public BowAlgorithm(File codebook) throws IOException, URISyntaxException {
        System.out.println("codebook file: " + codebook);

        Map<Feature.Type, List<double[]>> codebookMap = Collections.emptyMap();
        Map<Feature.Type, DistanceMeasure> distanceMeasures = new EnumMap<>(Feature.Type.class);
        if (codebook != null) {
            codebookMap = this.loadCodeBookMat(codebook);
            Map<Feature.Type, RealMatrix> mMatrixMap = this.loadKissmeMat(codebook);
            for (Feature.Type type : Feature.Type.values()) {
                if (mMatrixMap.containsKey(type)) {
                    distanceMeasures.put(type, new MahalanobisDistance(mMatrixMap.get(type)));
                } else {
                    distanceMeasures.put(type, new EuclideanDistance());
                }
            }
        }


        this.spMethod = new SlicMethod(numSuperpixels, compactness);
        this.stripMethod = new StripMethod(ystep * 4, stripLength * 4, pstep * 4);
        this.featureManager = new FeatureManager();
        this.bowManager = new BowManager(
                new Bow(K, sigma, codebookMap, new Mean(), distanceMeasures),
                this.featureManager,
                false);

        // output parameters
        System.out.println(this);
    }

    private Map<Feature.Type, List<double[]>> loadCodeBookMat(File mat) {
        try {
            MatFileReader reader = new MatFileReader(mat);

            Map<Feature.Type, List<double[]>> codebooks = new EnumMap<>(Feature.Type.class);
            for (Feature.Type type : EnumSet.allOf(Feature.Type.class)) {
                MLNumericArray ml = (MLNumericArray) reader.getMLArray("codebook_" + type);
                if (ml == null) {
                    ml = (MLNumericArray) reader.getMLArray("" + type);
                }
                if (ml == null) {
                    System.err.println("codebook not found, type=" + type);
                    continue;
                }

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

    private Map<Feature.Type, RealMatrix> loadKissmeMat(File mat) {
        try {
            MatFileReader reader = new MatFileReader(mat);

            Map<Feature.Type, RealMatrix> mMatrixMap = new EnumMap<>(Feature.Type.class);
            for (Feature.Type type : EnumSet.allOf(Feature.Type.class)) {
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

    @Override
    public List<MLArray> train(List<BowImage> bowImages) {
        bowImages.parallelStream().forEach(bowImage -> {
            if (bowImage.sp4 == null) {
                bowImage.sp4 = spMethod.generate(bowImage.image4);
            }
            if (bowImage.strip4 == null) {
                bowImage.strip4 = stripMethod.strip(bowImage.sp4, bowImage.mask4);
            }
        });

        List<MLArray> matrixes = new ArrayList<>();

        for (Feature.Type type : types) {
            bowImages.parallelStream().forEach(bowImage -> {
                System.out.println(type + ": pair=" + bowImage.camPair + ", id=" + bowImage.id + ", cam=" + bowImage.cam);
                featureManager.feature(bowImage, type);
            });

            RealMatrix M = kissMe.apply(bowImages, type);

            List<double[]> feature = bowImages.stream().flatMap(bowImage -> {
                List<double[]> features = new ArrayList<>();

                Set<Integer> nSuperPixels = new HashSet<>();
                for (Strip strip : bowImage.strip4) {
                    for (int n : strip.superPixels) {
                        if (nSuperPixels.contains(n)) {
                            break;
                        }
                        nSuperPixels.add(n);

                        features.add(bowImage.sp4[n].features.get(type));
                    }
                }
                return features.stream();
            }).collect(Collectors.toList());

            System.out.println("kmeans: " + type);
            ParallelKMeansPlusPlusClusterer<DoublePoint> clusterer = new ParallelKMeansPlusPlusClusterer<>(
                    350,
                    100,
                    new MahalanobisDistance(M),
                    new JDKRandomGenerator(),
                    KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT);
            List<double[]> codeBook = CodeBook.codebook(clusterer, feature);

            matrixes.add(MatrixUtils.to("codebook_" + type, codeBook));
            matrixes.add(new MLDouble("M_" + type, M.getData()));
        }

        return matrixes;
    }

    @Override
    public double[] test(BowImage bowImage) {
        if (bowImage.sp4 == null) {
            bowImage.sp4 = spMethod.generate(bowImage.image4);
        }
        if (bowImage.strip4 == null) {
            bowImage.strip4 = stripMethod.strip(bowImage.sp4, bowImage.mask4);
        }

        bowManager.bow(bowImage);

        double[][] hists = new double[types.length][];
        for (int i = 0; i < types.length; i++) {
            hists[i] = bowImage.hist.get(types[i]);
        }
        double[] hist = Doubles.concat(hists);

        // normalize
        double sum = Math.sqrt(new SumOfSquares().evaluate(hist));
        for (int i = 0; i < hist.length; i++) {
            hist[i] = hist[i] / sum;
        }
        return hist;
    }

    @Override
    public String toString() {
        return "BowAlgorithm{" +
                "spMethod=" + spMethod +
                ", stripMethod=" + stripMethod +
                ", featureManager=" + featureManager +
                ", bowManager=" + bowManager +
                '}';
    }
}
