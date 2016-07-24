package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Market1501;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.util.MatrixUtils;
import com.jmatio.io.MatFileWriter;
import org.apache.commons.math3.util.Pair;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author fanliwen
 */
public class Market1501App extends App {

    private static final File queryCamFolder = new File("/data/reid/market1501/dataset/query");
    private static final File queryMaskFolder = new File("/data/reid/market1501/mask/query");

    private static final File testCamFolder = new File("/data/reid/market1501/dataset/bounding_box_test");
    private static final File testMaskFolder = new File("/data/reid/market1501/mask/bounding_box_test");

    private Market1501App() throws IOException, URISyntaxException, ClassNotFoundException {
        super();
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

        System.out.println("hist fusion");
        List<double[]> queryHist = (List<double[]>) app.fusion(queryBowImages, types);
        List<double[]> testHist = (List<double[]>) app.fusion(testBowImages, types);

        // write to mat
        System.out.println("write hist to mat");
        new MatFileWriter().write(
                "market1501_hist_" + numSuperpixels + "_" + compactness + ".mat",
                Arrays.asList(
                        MatrixUtils.to("Hist_query", queryHist),
                        MatrixUtils.to("Hist_test", testHist)
                )
        );

        List<Pair<Integer, Integer>> queryIdAndCam = app.idAndCam(queryCamFolder);
        List<Pair<Integer, Integer>> testIdAndCam = app.idAndCam(testCamFolder);

        // descriptor level
        System.out.println("calculate score");
        Pair<Double, double[]> mapAndCmc = new Market1501().eval(queryHist, queryIdAndCam, testHist, testIdAndCam);
        System.out.println("map=" + mapAndCmc.getFirst() + ", precision=" + Arrays.toString(mapAndCmc.getSecond()));

        // seperate
        for (Feature.Type type : types) {
            queryHist = (List<double[]>) app.fusion(queryBowImages, new Feature.Type[]{type});
            testHist = (List<double[]>) app.fusion(testBowImages, new Feature.Type[]{type});
            mapAndCmc = new Market1501().eval(queryHist, queryIdAndCam, testHist, testIdAndCam);
            System.out.println("map=" + mapAndCmc.getFirst() + ", precision=" + Arrays.toString(mapAndCmc.getSecond()));
        }
    }
}
