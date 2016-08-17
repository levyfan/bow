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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Market1501TrainingApp extends App {

    private static final File training = new File("/data/reid/market1501/dataset/bounding_box_train");

    private static final File mask = new File("/data/reid/market1501/mask/bounding_box_train");

    private KissMe kissMe = new KissMe();

    private Market1501TrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
    }

    private List<BowImage> featureTraining(File folder, File maskFolder, Feature.Type type) throws IOException {
        File[] files = folder.listFiles(filter);

        return Lists.newArrayList(files)
                .parallelStream()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        String maskFileName = file.getName().replace("jpg", "bmp");
                        BufferedImage mask = ImageIO.read(new File(maskFolder, maskFileName));

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
        Market1501TrainingApp app = new Market1501TrainingApp();

        for (Feature.Type type : App.types) {
            // extract feature
            List<BowImage> bowImages = app.featureTraining(training, mask, type);

            // KissMe
            RealMatrix M = app.kissMe.kissMe(bowImages, type);

            // feature fusion
            Iterable<double[]> feature = app.fusion(bowImages, new Feature.Type[]{type});

            // clear bowImages to release memory
            bowImages.clear();

            // kmeans
//            ElkanKmeansPlusPlusClusterer<DoublePoint> clusterer = new ElkanKmeansPlusPlusClusterer<>(
//                    new MahalanobisDistance(M),
//                    codeBookSize,
//                    100);
            ParallelKMeansPlusPlusClusterer<DoublePoint> clusterer = new ParallelKMeansPlusPlusClusterer<>(
                    codeBookSize,
                    100,
                    new MahalanobisDistance(M),
                    new JDKRandomGenerator(),
                    KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT);
            List<DoublePoint> points = new ArrayList<>();
            for (double[] point : feature) {
                points.add(new DoublePoint(point));
            }
            System.out.println("kmeans: " + type);

            List<double[]> codeBook = clusterer.cluster(points)
                    .stream()
                    .map(centroidCluster -> centroidCluster.getCenter().getPoint())
                    .collect(Collectors.toList());

            System.out.print("start translate to mat");
            new MatFileWriter().write(
                    "Market_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                    Lists.newArrayList(
                            MatrixUtils.to("codebook_" + type, codeBook),
                            new MLDouble("M_" + type, M.getData())
                    )
            );
        }
    }
}
