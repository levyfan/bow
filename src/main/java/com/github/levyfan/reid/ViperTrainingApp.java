package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.ml.KissMe;
import com.github.levyfan.reid.ml.MahalanobisDistance;
import com.github.levyfan.reid.sp.SuperPixel;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.ParallelKMeansPlusPlusClusterer;
import org.apache.commons.math3.random.JDKRandomGenerator;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class ViperTrainingApp extends App {

    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    private KissMe kissMe = new KissMe();

    private ViperTrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
    }

    private List<BowImage> featureTraining(File folder, File maskFolder, Feature.Type type) throws IOException {
        File[] files = folder.listFiles(filter);

        return Lists.newArrayList(files)
                .parallelStream()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        String maskFileName = file.getName();
                        BufferedImage mask = ImageIO.read(new File(maskFolder, "mask_" + maskFileName));

                        BowImage bowImage = new BowImage(
                                spMethod, stripMethod, image, mask);
                        featureManager.feature(bowImage, type);

                        bowImage.id = file.getName().split("_")[0];
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    @Override
    Iterable<double[]> fusion(List<BowImage> bowImages, Feature.Type[] types) {
        List<List<double[]>> list = bowImages.stream().map(bowImage -> {
            List<double[]> features = new ArrayList<>();
            Set<Integer> nSuperPixels = new HashSet<>();
            for (Strip strip : bowImage.strip4) {
                for (int n : strip.superPixels) {
                    if (nSuperPixels.contains(n)) {
                        break;
                    }
                    nSuperPixels.add(n);

                    SuperPixel superPixel = bowImage.sp4[n];
                    double[][] fusion = new double[types.length][];
                    for (int i = 0; i < types.length; i++) {
                        fusion[i] = superPixel.features.get(types[i]);
                    }
                    features.add(Doubles.concat(fusion));
                }
            }
            return features;
        }).collect(Collectors.toList());

        return Iterables.concat(list);
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        ViperTrainingApp app = new ViperTrainingApp();

        for (Feature.Type type : App.types) {
            // extract feature
            List<BowImage> bowImages = app.featureTraining(testingA, maskA, type);
            bowImages.addAll(app.featureTraining(testingB, maskB, type));

            // KissMe
            RealMatrix M = app.kissMe.kissMe(bowImages, type);

            // feature fusion
            Iterable<double[]> feature = app.fusion(bowImages, new Feature.Type[]{type});

            // clear bowImages to release memory
            bowImages.clear();

            // kmeans
            System.out.println("kmeans: " + type);
            ParallelKMeansPlusPlusClusterer<DoublePoint> clusterer = new ParallelKMeansPlusPlusClusterer<>(
                    codeBookSize,
                    100,
                    new MahalanobisDistance(M),
                    new JDKRandomGenerator(),
                    KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT);
            List<double[]> codeBook = app.codeBook.codebook(clusterer, feature);

            System.out.print("start translate to mat");
            new MatFileWriter().write(
                    "Viper_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                    Lists.newArrayList(
                            MatrixUtils.to("codebook_" + type, codeBook),
                            new MLDouble("M_" + type, M.getData())
                    )
            );
        }
    }
}
