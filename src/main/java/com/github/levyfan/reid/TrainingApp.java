package com.github.levyfan.reid;

import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Lists;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.ParallelKMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.random.JDKRandomGenerator;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class TrainingApp extends App {

    private static final File training = new File("/data/reid/TUDpositive");
    private static final File mask = new File("/data/reid/TUDpositive_mask");

    private TrainingApp() throws IOException, URISyntaxException, ClassNotFoundException {
        super(null);
    }

    private List<BowImage> featureTraining(File folder, File maskFolder) throws IOException {
        File[] files = folder.listFiles(filter);

        return Lists.newArrayList(files)
                .parallelStream()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        String maskFileName = file.getName().replace("png", "bmp");
                        BufferedImage mask = ImageIO.read(new File(maskFolder, maskFileName));

                        BowImage bowImage = new BowImage(
                                spMethod, stripMethod, image, mask);
                        featureManager.feature(bowImage);
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException, URISyntaxException, ClassNotFoundException {
        TrainingApp app = new TrainingApp();

        // extract feature
        List<BowImage> bowImages = app.featureTraining(training, mask);

        Map<Feature.Type, Iterable<double[]>> featureMap = new EnumMap<>(Feature.Type.class);
        featureMap.put(Feature.Type.HSV, app.fusionFeatures(bowImages, new Feature.Type[]{Feature.Type.HSV}));
        featureMap.put(Feature.Type.CN, app.fusionFeatures(bowImages, new Feature.Type[]{Feature.Type.CN}));
        featureMap.put(Feature.Type.HOG, app.fusionFeatures(bowImages, new Feature.Type[]{Feature.Type.HOG}));
        featureMap.put(Feature.Type.SILTP, app.fusionFeatures(bowImages, new Feature.Type[]{Feature.Type.SILTP}));

        // clear bowImages to release memory
        bowImages.clear();

        Map<Feature.Type, MLArray> codebookMap = new EnumMap<>(Feature.Type.class);
        // training
        for (Feature.Type type : App.types) {
            Iterable<double[]> feature = featureMap.get(type);

            System.out.println("kmeans: " + type);
            ParallelKMeansPlusPlusClusterer<DoublePoint> clusterer = new ParallelKMeansPlusPlusClusterer<>(
                    codeBookSize,
                    100,
                    new EuclideanDistance(),
                    new JDKRandomGenerator(),
                    KMeansPlusPlusClusterer.EmptyClusterStrategy.FARTHEST_POINT);
            List<double[]> codeBook = app.codeBook.codebook(clusterer, feature);

            codebookMap.put(type, MatrixUtils.to("codebook_" + type, codeBook));
        }

        System.out.println("start translate to mat");
        new MatFileWriter().write(
                "TUD_positive_" + numSuperpixels + "_" + compactness + ".mat",
                codebookMap.values());
    }
}
