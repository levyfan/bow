package com.github.levyfan.reid.dataset;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.algorithm.Algorithm;
import com.github.levyfan.reid.algorithm.BowAlgorithm;
import com.google.common.collect.*;
import com.google.common.io.PatternFilenameFilter;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileType;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Cuhk03 {

    private ListMultimap[] testIndexes;

    private boolean detected;

    private static final File DETECTED = new File("/data/reid/cuhk03/detected");
    private static final File LABELED = new File("/data/reid/cuhk03/labeled");
    private static final File MASK_DETECTED = new File("/data/reid/cuhk03/mask/detected");
    private static final File MASK_LABELED = new File("/data/reid/cuhk03/mask/labeled");

    private static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    private List<BowImage> images;

    private Cuhk03(boolean detected) throws IOException {
        try (InputStream stream = this.getClass().getResourceAsStream("/testsets.mat")) {
            MatFileReader reader = new MatFileReader(stream, MatFileType.Regular);
            MLCell testsets = (MLCell) reader.getMLArray("testsets");

            testIndexes = new ListMultimap[testsets.getM()];
            for (int m = 0; m < testsets.getM(); m++) {
                testIndexes[m] = ArrayListMultimap.create();

                MLNumericArray testset = (MLNumericArray) testsets.get(m, 0);
                for (int n = 0; n < testset.getM(); n++) {
                    testIndexes[m].put(testset.get(n, 0).intValue(), testset.get(n, 1).intValue());
                }
            }
            System.out.println("trials=" + testIndexes.length);
        }

        File folder;
        File maskFolder;
        if (detected) {
            folder = DETECTED;
            maskFolder = MASK_DETECTED;
        } else {
            folder = LABELED;
            maskFolder = MASK_LABELED;
        }

        File[] files = folder.listFiles(filter);
        Arrays.sort(files);
        images = Arrays.stream(files).map(file -> {
            try {
                BufferedImage image = ImageIO.read(file);
                BufferedImage mask = ImageIO.read(new File(maskFolder, file.getName()));

                String[] splits = file.getName().substring(0, file.getName().indexOf('.')).split("_");
                BowImage bowImage = new BowImage(image, mask);
                bowImage.id = splits[1];
                bowImage.cam = (Integer.valueOf(splits[2]) > 5 ? "2" : "1");
                bowImage.camPair = splits[0];

                if (bowImage.image == null
                        || bowImage.image4 == null
                        || bowImage.mask == null
                        || bowImage.mask4 == null) {
                    System.err.println(file.getPath());
                }
                return bowImage;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }).collect(Collectors.toList());

        this.detected = detected;
    }

    public void train(Algorithm algorithm) throws IOException {
        for (int trial = 0; trial < testIndexes.length; trial ++) {
            for (int pair = 1; pair <= 3; pair ++) {
                System.out.println("trial:" + trial + ", pair:" + pair);
                Set<Integer> testIndex = Sets.newHashSet(testIndexes[trial].get(pair));

                int finalPair = pair;
                List<BowImage> trainList = images.stream().filter(image ->
                        finalPair == Integer.valueOf(image.camPair)
                                && !testIndex.contains(Integer.valueOf(image.id))
                ).collect(Collectors.toList());

                List<MLArray> matrixes = algorithm.train(trainList);
                if (detected) {
                    new MatFileWriter().write("cuhk03_detected_" + trial + "_" + pair + ".mat", matrixes);
                } else {
                    new MatFileWriter().write("cuhk03_labeled_" + trial + "_" + pair + ".mat", matrixes);
                }
            }
        }
    }

    public double[] test(Algorithm algorithm) {
        double[] cmc = new double[images.size()];

        for (int trial = 0; trial < testIndexes.length; trial ++) {
            double[] t = test(algorithm, trial);
            System.out.println(String.format(
                    "trial %d: r1=%f, r5=%f, r10=%f, r20=%f", trial, t[0], t[4], t[9], t[19]));

            for (int i = 0; i < t.length; i ++) {
                cmc[i] += t[i];
            }
        }

        for (int i = 0; i < cmc.length; i ++) {
            cmc[i] = cmc[i] / testIndexes.length;
        }

        System.out.println(String.format(
                "r1=%f, r5=%f, r10=%f, r20=%f", cmc[0], cmc[4], cmc[9], cmc[19]));
        return cmc;
    }

    public double[] test(Algorithm algorithm, int trial) {
        System.out.println("trial: " + trial);
        ListMultimap testIndex = testIndexes[trial];

        List<double[]> descriptors = images.parallelStream()
                .map(bowImage -> {
                    if (testIndex.get(Integer.valueOf(bowImage.camPair)).contains(Integer.valueOf(bowImage.id))) {
                        System.out.println(
                                "pair: " + bowImage.camPair + ", cam:" + bowImage.cam + ", id: " + bowImage.id);
                        return algorithm.test(bowImage);
                    } else {
                        return null;
                    }
                }).collect(Collectors.toList());

        Table<Integer, Integer, Double> distance = HashBasedTable.create();
        for (int i = 0; i < descriptors.size(); i ++) {
            if (descriptors.get(i) != null && "1".equals(images.get(i).cam)) {
                for (int j = 0; j < descriptors.size(); j++) {
                    if (descriptors.get(j) != null && "2".equals(images.get(j).cam)) {
                        double d = new EuclideanDistance().compute(descriptors.get(i), descriptors.get(j));
                        distance.put(i, j, d);
                    }
                }
            }
        }

        // cmc
        int[] ranks = new int[descriptors.size()];

        // 1 as gallery, 2 as probe
        for (int j : distance.columnKeySet()) {
            Map<Integer, Double> score = distance.column(j);

            List<Integer> is = score.entrySet().stream()
                    .sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            for (int rank = 0; rank < is.size(); rank ++) {
                int i = is.get(rank);
                if (images.get(i).id.equals(images.get(j).id)
                        && images.get(i).camPair.equals(images.get(j).camPair)) {
                    ranks[rank] ++;
                    break;
                }
            }
        }

        // 2 as gallery, 1 as probe
        for (int i : distance.rowKeySet()) {
            Map<Integer, Double> score = distance.row(i);

            List<Integer> js = score.entrySet().stream()
                    .sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            for (int rank = 0; rank < js.size(); rank ++) {
                int j = js.get(rank);
                if (images.get(i).id.equals(images.get(j).id)
                        && images.get(i).camPair.equals(images.get(j).camPair)) {
                    ranks[rank] ++;
                    break;
                }
            }
        }

        double[] cmc = new double[ranks.length];
        for (int i = 0; i < ranks.length; i ++) {
            cmc[i] = ((double) ranks[i]) / (distance.rowKeySet().size() + distance.columnKeySet().size());
        }

        Sum summary = new Sum();
        for (int i = 0; i < cmc.length; i ++) {
            summary.increment(cmc[i]);
            cmc[i] = summary.getResult();
        }
        return cmc;
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        Cuhk03 cuhk03 = new Cuhk03(true);
        BowAlgorithm algorithm = new BowAlgorithm(new File(args[0]));

        try {
            // train
            // cuhk03.train(algorithm);

            // test
            cuhk03.test(algorithm);

        } catch (Throwable throwable) {
            System.err.println("Uncaught exception - " + throwable.getMessage());
            throwable.printStackTrace(System.err);
        }
    }
}
