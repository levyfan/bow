package com.github.levyfan.reid.dataset;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.algorithm.Algorithm;
import com.github.levyfan.reid.algorithm.BowAlgorithm;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Sets;
import com.google.common.io.PatternFilenameFilter;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileType;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLNumericArray;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
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

    public static void main(String[] args) throws IOException, URISyntaxException {
        Cuhk03 cuhk03 = new Cuhk03(true);
        BowAlgorithm algorithm = new BowAlgorithm(null);

        cuhk03.train(algorithm);
    }
}
