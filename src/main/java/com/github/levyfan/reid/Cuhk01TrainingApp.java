package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Cuhk01;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.ml.KissMe;
import com.github.levyfan.reid.ml.MahalanobisDistance;
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

/**
 * @author fanliwen
 */
public class Cuhk01TrainingApp extends App {

    private static final File testing = new File("/data/reid/cuhk01_resize/campus");
    private static final File mask = new File("/data/reid/cuhk01_resize/mask");

    private KissMe kissMe = new KissMe();
    private Cuhk01 cuhk01 = new Cuhk01();

    private Cuhk01TrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
        super(null);
    }

    private List<BowImage> featureTraining(
            File folder,
            File maskFolder,
            Feature.Type type,
            Set<Integer> testIds) throws IOException {
        File[] files = folder.listFiles(filter);
        Arrays.sort(files);
        System.out.println(Arrays.toString(files));

        return Arrays.stream(files)
                .filter(file -> !testIds.contains(Integer.valueOf(file.getName().substring(0, 4))))
                .parallel()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        String maskFileName = file.getName();
                        BufferedImage mask = ImageIO.read(new File(maskFolder, maskFileName));

                        BowImage bowImage = new BowImage(
                                spMethod, stripMethod, image, mask);
                        featureManager.feature(bowImage, type);

                        bowImage.id = file.getName().substring(0, 4);
                        bowImage.cam = String.valueOf(
                                (Integer.valueOf(file.getName().substring(4, 7)) + 1) / 2
                        );
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        Cuhk01TrainingApp app = new Cuhk01TrainingApp();

        int nloop = app.cuhk01.select.getColumnDimension();
        for (int loop = 0; loop < nloop; loop++) {
            System.out.println("loop: " + loop);
            Set<Integer> testIds = new HashSet<>(Ints.asList(
                    DoubleStream.of(app.cuhk01.select.getColumn(loop)).mapToInt(s -> (int) s).toArray()));

            for (Feature.Type type : App.types) {
                // extract feature
                List<BowImage> bowImages = app.featureTraining(testing, mask, type, testIds);

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
                            "Cuhk01_kissme_loop" + loop + "_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                            Lists.newArrayList(
                                    MatrixUtils.to("codebook_" + type, codeBook),
                                    new MLDouble("M_" + type, M.getData())
                            )
                    );
                } catch (Exception e) {
                    new MatFileWriter().write(
                            "Cuhk01_error_kissme_loop" + loop + "_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
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
