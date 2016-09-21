package com.github.levyfan.reid;

import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.sp.SuperPixel;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;

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

    @Override
    Iterable<double[]> fusionHists(List<BowImage> bowImages, Feature.Type[] types) {
        List<List<double[]>> list = bowImages.stream().map(bowImage -> {
            List<double[]> features = new ArrayList<>();
            for (SuperPixel superPixel : bowImage.sp4) {
                double[][] fusion = new double[types.length][];
                for (int i = 0; i < types.length; i++) {
                    fusion[i] = superPixel.features.get(types[i]);
                }
                features.add(Doubles.concat(fusion));
            }
            return features;
        }).collect(Collectors.toList());

        return Iterables.concat(list);
    }

    Map<Integer, Collection<double[]>> fusion(List<BowImage> bowImages) {
        ListMultimap<Integer, double[]> multimap = ArrayListMultimap.create();
        for (BowImage bowImage : bowImages) {
            Set<Integer> nSuperPixels = new HashSet<>();
            for (Strip strip : bowImage.strip4) {
                for (int n : strip.superPixels) {
                    if (nSuperPixels.contains(n)) {
                        break;
                    }
                    nSuperPixels.add(n);

                    SuperPixel superPixel = bowImage.sp4[n];
                    double[] feature = Doubles.concat(
                            superPixel.features.get(App.types[0]),
                            superPixel.features.get(App.types[1]),
                            superPixel.features.get(App.types[2]),
                            superPixel.features.get(App.types[3]));

                    multimap.put(strip.index, feature);
                }
            }
        }
        return multimap.asMap();
    }

    public static void main(String[] args) throws IOException, URISyntaxException, ClassNotFoundException {
        TrainingApp app = new TrainingApp();

        // extract feature
        List<BowImage> bowImages = app.featureTraining(training, mask);

        // word level fusion
//        Map<Feature.Type, Iterable<double[]>> featureMap = new EnumMap<>(Feature.Type.class);
//        featureMap.put(Feature.Type.HSV, app.fusion(bowImages, new Feature.Type[]{Feature.Type.HSV}));
//        featureMap.put(Feature.Type.CN, app.fusion(bowImages, new Feature.Type[]{Feature.Type.CN}));
//        featureMap.put(Feature.Type.HOG, app.fusion(bowImages, new Feature.Type[]{Feature.Type.HOG}));
//        featureMap.put(Feature.Type.SILTP, app.fusion(bowImages, new Feature.Type[]{Feature.Type.SILTP}));

        Map<Integer, Collection<double[]>> featureMap = app.fusion(bowImages);

        // clear bowImages to release memory
        bowImages.clear();

        // save feature map to mat file
        System.out.print("start translate to mat");
//        new MatFileWriter().write(
//                "TUDpositive_feature_" + numSuperpixels + "_" + compactness + ".mat",
//                Lists.newArrayList(
//                        MatrixUtils.to("hsv", featureMap.get(Feature.Type.HSV)),
//                        MatrixUtils.to("cn", featureMap.get(Feature.Type.CN)),
//                        MatrixUtils.to("hog", featureMap.get(Feature.Type.HOG)),
//                        MatrixUtils.to("siltp", featureMap.get(Feature.Type.SILTP))
//                )
//        );
        new MatFileWriter().write(
                "TUDpositive_parse_" + numSuperpixels + "_" + compactness + ".mat",
                Lists.newArrayList(
                        MatrixUtils.to("p0", featureMap.get(0)),
                        MatrixUtils.to("p1", featureMap.get(1)),
                        MatrixUtils.to("p2", featureMap.get(2)),
                        MatrixUtils.to("p3", featureMap.get(3)),
                        MatrixUtils.to("p4", featureMap.get(4))
                )
        );
    }
}
