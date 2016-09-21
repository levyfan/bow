package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Prid450s;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.ml.KissMe;
import com.github.levyfan.reid.ml.MahalanobisDistance;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Lists;
import com.google.common.io.PatternFilenameFilter;
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
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class Prid450sTrainingApp extends App {

    private static final File testingA = new File("/data/reid/prid_450s_resize/cam_a");
    private static final File testingB = new File("/data/reid/prid_450s_resize/cam_b");

    private KissMe kissMe = new KissMe();
    private Prid450s prid450s = new Prid450s();

    private Prid450sTrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
        super(null);
    }

    private List<BowImage> featureTraining(
            File folder,
            File maskFolder,
            Feature.Type type,
            Set<Integer> testIndex) throws IOException {
        File[] files = folder.listFiles(new PatternFilenameFilter("img_.*\\.bmp"));
        Arrays.sort(files);
        System.out.println(Arrays.toString(files));

        return IntStream.range(0, files.length).filter(i -> !testIndex.contains(i))
                .parallel()
                .mapToObj(i -> {
                    try {
                        System.out.println(files[i].getName());
                        BufferedImage image = ImageIO.read(files[i]);

                        String maskFileName = files[i].getName().replace("img", "man");
                        BufferedImage mask = ImageIO.read(new File(maskFolder, maskFileName));

                        BowImage bowImage = new BowImage(
                                spMethod, stripMethod, image, mask);
                        featureManager.feature(bowImage, type);

                        bowImage.id = files[i].getName().split("\\.")[0].split("_")[1];
                        bowImage.cam = folder.getPath();
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        Prid450sTrainingApp app = new Prid450sTrainingApp();

        int nloop = app.prid450s.select.getColumnDimension();
        for (int loop = 0; loop < nloop; loop++) {
            System.out.println("loop: " + loop);
            Set<Integer> testIndex = new HashSet<>(Ints.asList(
                    DoubleStream.of(app.prid450s.select.getColumn(loop)).mapToInt(s -> (int) (s-1)).toArray()));

            for (Feature.Type type : App.types) {
                // extract feature
                List<BowImage> bowImages = app.featureTraining(testingA, testingA, type, testIndex);
                bowImages.addAll(app.featureTraining(testingB, testingB, type, testIndex));

                // KissMe
                RealMatrix M = app.kissMe.apply(bowImages, type);

                // feature fusion
                Iterable<double[]> feature = app.fusionFeatures(bowImages, new Feature.Type[]{type});

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
                        "Prid450s_loop" + loop + "_" + type + "_" + numSuperpixels + "_" + compactness + ".mat",
                        Lists.newArrayList(
                                MatrixUtils.to("codebook_" + type, codeBook),
                                new MLDouble("M_" + type, M.getData())
                        )
                );
            }
        }

        System.out.println("done");
    }
}
