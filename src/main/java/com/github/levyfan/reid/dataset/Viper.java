package com.github.levyfan.reid.dataset;

import com.github.levyfan.reid.algorithm.Algorithm;
import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.algorithm.BowAlgorithm;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import com.google.common.io.PatternFilenameFilter;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileType;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.summary.Sum;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class Viper {

    private Set<Integer>[] testIndexes;

    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    private static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    private List<BowImage> a;
    private List<BowImage> b;

    private Viper() throws URISyntaxException, IOException {
        try (InputStream stream = this.getClass().getResourceAsStream("/randselect10.mat")) {
            MatFileReader reader = new MatFileReader(stream, MatFileType.Regular);
            MLNumericArray selectsample = (MLNumericArray) reader.getMLArray("selectsample");

            testIndexes = new Set[selectsample.getN()];
            for (int n = 0; n < selectsample.getN(); n++) {
                testIndexes[n] = new HashSet<>();
                for (int m = 0; m < selectsample.getM(); m++) {
                    testIndexes[n].add(selectsample.get(m, n).intValue() - 1);
                }
            }
            System.out.println("trials=" + testIndexes.length);
        }

        File[] filesA = testingA.listFiles(filter);
        Arrays.sort(filesA);
        a = Arrays.stream(filesA).map(file -> {
            try {
                BufferedImage image = ImageIO.read(file);
                BufferedImage mask = ImageIO.read(new File(maskA, "mask_" + file.getName().replace("jpg", "bmp")));

                BowImage bowImage = new BowImage(image, mask);
                bowImage.id = file.getName().split("_")[0];
                bowImage.cam = "a";
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

        File[] filesB = testingB.listFiles(filter);
        Arrays.sort(filesB);
        b = Arrays.stream(filesB).map(file -> {
            try {
                BufferedImage image = ImageIO.read(file);
                BufferedImage mask = ImageIO.read(new File(maskB, "mask_" + file.getName().replace("jpg", "bmp")));

                BowImage bowImage = new BowImage(image, mask);
                bowImage.id = file.getName().split("_")[0];
                bowImage.cam = "b";
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
    }


    public void train(Algorithm algorithm, int trial) {
        Set<Integer> testIndex = testIndexes[trial];

        List<BowImage> trainList = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            if (!testIndex.contains(i)) {
                trainList.add(a.get(i));
            }
        }
        for (int j = 0; j < b.size(); j++) {
            if (!testIndex.contains(j)) {
                trainList.add(b.get(j));
            }
        }

        algorithm.train(trainList);
    }

    public double[] test(Algorithm algorithm) {
        double[] cmc = new double[Math.max(a.size(), b.size())];

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
        Set<Integer> testIndex = testIndexes[trial];

        List<double[]> A = IntStream.range(0, a.size()).parallel().mapToObj(index -> {
            if (testIndex.contains(index)) {
                System.out.println("cam: " + a.get(index).cam + ", id:" + a.get(index).id);
                return algorithm.test(a.get(index));
            } else {
                return null;
            }
        }).collect(Collectors.toList());

        List<double[]> B = IntStream.range(0, b.size()).parallel().mapToObj(index -> {
            if (testIndex.contains(index)) {
                System.out.println("cam: " + b.get(index).cam + ", id:" + b.get(index).id);
                return algorithm.test(b.get(index));
            } else {
                return null;
            }
        }).collect(Collectors.toList());

        Table<Integer, Integer, Double> distance = HashBasedTable.create();
        for (int i = 0; i < A.size(); i ++) {
            for (int j = 0; j < B.size(); j++) {
                if (A.get(i) != null && B.get(j) != null) {
                    double d = new EuclideanDistance().compute(A.get(i), B.get(j));
                    distance.put(i, j, d);
                }
            }
        }

        // cmc
        int[] ranks = new int[Math.max(a.size(), b.size())];

        // A as gallery, B as probe
        for (int j : distance.columnKeySet()) {
            Map<Integer, Double> score = distance.column(j);

            List<Integer> is = score.entrySet().stream()
                    .sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            for (int rank = 0; rank < is.size(); rank ++) {
                int i = is.get(rank);
                if (a.get(i).id.equals(b.get(j).id)) {
                    ranks[rank] ++;
                    break;
                }
            }
        }

        // A as probe, B as gallery
        for (int i : distance.rowKeySet()) {
            Map<Integer, Double> score = distance.row(i);

            List<Integer> js = score.entrySet().stream()
                    .sorted((e1, e2) -> Double.compare(e1.getValue(), e2.getValue()))
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toList());

            for (int rank = 0; rank < js.size(); rank ++) {
                int j = js.get(rank);
                if (a.get(i).id.equals(b.get(j).id)) {
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
        Viper viper = new Viper();
        BowAlgorithm algorithm = new BowAlgorithm(new File(args[0]));

        viper.test(algorithm);
    }
}
