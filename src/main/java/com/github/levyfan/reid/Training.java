package com.github.levyfan.reid;

import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.util.Pair;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Training extends App {

    private Training() throws IOException, URISyntaxException, ClassNotFoundException {
        super();
    }

    private List<BowImage> featureTraining(File folder) throws IOException {
        File[] files = folder.listFiles(filter);

        return Lists.newArrayList(files)
                .parallelStream()
                .map(file -> {
                    try {
                        System.out.println(file.getName());
                        BufferedImage image = ImageIO.read(file);

                        BowImage bowImage = new BowImage(spMethod, null, image, null);
                        featureManager.feature(bowImage);
                        return bowImage;
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    private Iterable<double[]> fusion(List<BowImage> bowImages, Feature.Type[] types) {
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

    private Map<Feature.Type, List<double[]>> codeBookTraining(
            File folder, Map<Feature.Type, Iterable<double[]>> featureMap) throws IOException {
        Map<Feature.Type, List<double[]>> books = featureMap.entrySet()
                .parallelStream()
                .map(entry -> {
                    System.out.println("codebook gen start " + entry.getKey());

                    List<double[]> words = codeBook.codebook(
                            entry.getValue(), entry.getKey() == Feature.Type.ALL ? codeBookSize * 4 : codeBookSize);

                    System.out.println("codebook gen done " + entry.getKey());
                    return Pair.create(entry.getKey(), words);
                }).collect(Collectors.toMap(Pair::getFirst, Pair::getSecond));

        File dat = new File(folder, "codebook_wordlevel_fix_slic_" + numSuperpixels + "_" + compactness + ".dat");
        try (ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(dat))) {
            os.writeObject(books);
        }
        return books;
    }

    private static Double[] to(Iterable<double[]> iterable) {
        int size = Iterables.size(iterable) * Iterables.getFirst(iterable, null).length;

        List<Double> list = new ArrayList<>(size);
        for (double[] doubles : iterable) {
            list.addAll(Doubles.asList(doubles));
        }
        return list.toArray(new Double[0]);
    }

    public static void main(String[] args) throws IOException, URISyntaxException, ClassNotFoundException {
        Training app = new Training();

        // extract feature
        List<BowImage> bowImages = app.featureTraining(training);

        // word level fusion
        Map<Feature.Type, Iterable<double[]>> featureMap = new EnumMap<>(Feature.Type.class);
        featureMap.put(Feature.Type.HSV, app.fusion(bowImages, new Feature.Type[]{Feature.Type.HSV}));
        featureMap.put(Feature.Type.CN, app.fusion(bowImages, new Feature.Type[]{Feature.Type.CN}));
        featureMap.put(Feature.Type.HOG, app.fusion(bowImages, new Feature.Type[]{Feature.Type.HOG}));
        featureMap.put(Feature.Type.SILTP, app.fusion(bowImages, new Feature.Type[]{Feature.Type.SILTP}));
//        featureMap.put(Feature.Type.ALL, app.fusion(bowImages, types));

        int hsvLength = bowImages.get(0).sp4[0].features.get(Feature.Type.HSV).length;
        int cnLength = bowImages.get(0).sp4[0].features.get(Feature.Type.CN).length;
        int hogLength = bowImages.get(0).sp4[0].features.get(Feature.Type.HOG).length;
        int siltpLength = bowImages.get(0).sp4[0].features.get(Feature.Type.SILTP).length;
//        int allLength = hsvLength + cnLength + hogLength + siltpLength;

        // clear bowImages to release memory
        bowImages.clear();

        MLDouble hsv = new MLDouble("hsv", to(featureMap.get(Feature.Type.HSV)), hsvLength);
        MLDouble cn = new MLDouble("cn", to(featureMap.get(Feature.Type.CN)), cnLength);
        MLDouble hog = new MLDouble("hog", to(featureMap.get(Feature.Type.HOG)), hogLength);
        MLDouble siltp = new MLDouble("siltp", to(featureMap.get(Feature.Type.SILTP)), siltpLength);
//        double[] all = Doubles.concat(Iterables.toArray(featureMap.get(Feature.Type.ALL), double[].class));
        featureMap.clear();

        // save feature map to mat file
        System.out.print("start translate to mat");
        List<MLArray> mlArrays = Lists.newArrayList(hsv, cn, hog, siltp
//                new MLDouble("all", Doubles.asList(all).toArray(new Double[0]), allLength)
        );
        System.out.print("start writing mat");
        new MatFileWriter().write(
                "TUDpositive_feature_" + numSuperpixels + "_" + compactness + ".mat",
                mlArrays);

        // code book training
//        Map<Feature.Type, List<double[]>> codebookMap = app.codeBookTraining(training, featureMap);
//        new MatFileWriter().write(
//                "codebook_wordlevel_fix_" + numSuperpixels + "_" + compactness + ".mat",
//                codebookMap.entrySet()
//                        .stream()
//                        .map(entry -> new MLDouble(entry.getKey().name(), entry.getValue().toArray(new double[0][])))
//                        .collect(Collectors.toList()));
    }
}
