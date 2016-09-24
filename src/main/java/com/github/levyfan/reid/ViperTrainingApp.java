package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Viper;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.ml.KissMe;
import com.github.levyfan.reid.ml.MahalanobisDistance;
import com.github.levyfan.reid.ml.Xqda;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;
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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class ViperTrainingApp extends App {

    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    private KissMe kissMe = new Xqda();
    private Viper viper = new Viper();

    private ViperTrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
        super(null);
    }

    private List<BowImage> featureTraining(
            File folder,
            File maskFolder,
            Feature.Type type,
            Set<Integer> testIndex) throws IOException {
        File[] files = folder.listFiles(filter);
        Arrays.sort(files);
        System.out.println(Arrays.toString(files));

        return IntStream.range(0, files.length).filter(i -> !testIndex.contains(i))
                .parallel()
                .mapToObj(i -> {
                    try {
                        System.out.println(files[i].getName());
                        BufferedImage image = ImageIO.read(files[i]);

                        String maskFileName = files[i].getName();
                        BufferedImage mask = ImageIO.read(new File(maskFolder, "mask_" + maskFileName));

                        BowImage bowImage = new BowImage(
                                spMethod, stripMethod, image, mask);
                        featureManager.feature(bowImage, type);

                        bowImage.id = files[i].getName().split("_")[0];
                        bowImage.cam = folder.getPath();
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        ViperTrainingApp app = new ViperTrainingApp();

        int nloop = app.viper.select.getColumnDimension();
        for (int loop = 0; loop < nloop; loop++) {
            System.out.println("loop: " + loop);
            Set<Integer> testIndex = new HashSet<>(Ints.asList(
                    DoubleStream.of(app.viper.select.getColumn(loop)).mapToInt(s -> (int) (s-1)).toArray()));

            for (Feature.Type type : App.types) {
                // extract feature
                List<BowImage> bowImages = app.featureTraining(testingA, maskA, type, testIndex);
                bowImages.addAll(app.featureTraining(testingB, maskB, type, testIndex));

                // KissMe
                RealMatrix M = app.kissMe.apply(bowImages, type);

                // feature fusion
                Iterable<double[]> feature = app.fusionFeatures(bowImages, new Feature.Type[]{type});

                // clear bowImages to release memory
                bowImages.clear();

                // kmeans
                try {
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
                            "Viper_xqda_loop" + loop + "_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                            Lists.newArrayList(
                                    MatrixUtils.to("codebook_" + type, codeBook),
                                    new MLDouble("M_" + type, M.getData())
                            )
                    );
                } catch (Exception e) {
                    new MatFileWriter().write(
                            "error_xqda_loop" + loop + "_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                            Lists.newArrayList(new MLDouble("M_" + type, M.getData()))
                    );

                    e.printStackTrace();
                    throw e;
                }
            }
        }

        System.out.println("done");
    }
}
