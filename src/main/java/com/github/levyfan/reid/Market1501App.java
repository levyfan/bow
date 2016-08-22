package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Market1501;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.base.Joiner;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;
import org.apache.commons.math3.util.Pair;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URISyntaxException;
import java.util.*;

/**
 * @author fanliwen
 */
public class Market1501App extends App {

    private static final File queryCamFolder = new File("/data/reid/market1501/dataset/query");
    private static final File queryMaskFolder = new File("/data/reid/market1501/mask/query");

    private static final File testCamFolder = new File("/data/reid/market1501/dataset/bounding_box_test");
    private static final File testMaskFolder = new File("/data/reid/market1501/mask/bounding_box_test");

    private static final File trainCamFolder = new File("/data/reid/market1501/dataset/bounding_box_train");
    private static final File trainMaskFolder = new File("/data/reid/market1501/mask/bounding_box_train");

    private Market1501App() throws IOException, URISyntaxException, ClassNotFoundException {
        super(new File("codebook_kissme_500_20.mat"));
    }

    private List<Pair<Integer, Integer>> idAndCam(File folder) {
        File[] files = folder.listFiles(filter);

        List<Pair<Integer, Integer>> pairs = new ArrayList<>(files.length);
        for (File file : files) {
            String[] strings = file.getName().split("_");
            int id = Integer.valueOf(strings[0]);
            int cam = Integer.valueOf(strings[1].substring(1, 2));

            pairs.add(Pair.create(id, cam));
        }
        return pairs;
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        Market1501App app = new Market1501App();

        List<BowImage> queryBowImages = app.generateHist(queryCamFolder, queryMaskFolder, "");
        List<BowImage> testBowImages = app.generateHist(testCamFolder, testMaskFolder, "");
//        List<BowImage> trainBowImages = app.generateHist(trainCamFolder, trainMaskFolder, "");

        System.out.println("hist fusion");
        List<double[]> queryHist = (List<double[]>) app.fusion(queryBowImages, types);
        List<double[]> testHist = (List<double[]>) app.fusion(testBowImages, types);
//        List<double[]> trainHist = (List<double[]>) app.fusion(trainBowImages, types);

        // write to mat/csv
        System.out.println("write query hist to mat");
        new MatFileWriter().write(
                "market1501_hist_query_" + numSuperpixels + "_" + compactness + "_" + pstep + ".mat",
                Collections.singleton(MatrixUtils.to("Hist_query", queryHist))
        );
        System.out.println("write test hist to cvs");
        try (FileOutputStream os = new FileOutputStream(
                "market1501_hist_test_" + numSuperpixels + "_" + compactness + "_" + pstep + ".csv");
             PrintWriter writer = new PrintWriter(os, true)) {
            for (double[] hist : testHist) {
                writer.println(Joiner.on(',').join(Doubles.asList(hist)));
            }
        }
//        System.out.println("write train hist to mat");
//        try (FileOutputStream os = new FileOutputStream(
//                "market1501_hist_train_" + numSuperpixels + "_" + compactness + "_" + pstep + ".csv");
//             PrintWriter writer = new PrintWriter(os, true)) {
//            for (double[] hist : trainHist) {
//                writer.println(Joiner.on(',').join(Doubles.asList(hist)));
//            }
//        }
    }
}
