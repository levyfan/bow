package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Viper;
import com.github.levyfan.reid.feature.Feature;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.stat.descriptive.summary.SumOfSquares;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class ViperParseApp extends App {

    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    private Viper viper = new Viper();

    private Map<Integer, List<double[]>> codebook;

    private ViperParseApp() throws IOException, URISyntaxException, ClassNotFoundException {
        super(new File("codebook_kissme_500_20.mat"));
        this.bowManager.getBow().getCodebooks().clear();

        this.codebook = this.loadCodeBook(new File("codebook.mat"));
    }

    private Map<Integer, List<double[]>> loadCodeBook(File mat) throws URISyntaxException, IOException {
        MatFileReader reader = new MatFileReader(mat);

        Map<Integer, List<double[]>> codebooks = new HashMap<>();
        for (int n = 0; n < 5; n++) {
            MLNumericArray ml = (MLNumericArray) reader.getMLArray("codebook_p" + n);

            List<double[]> codebook = new ArrayList<>();
            for (int i = 0; i < ml.getM(); i++) {
                double[] word = new double[ml.getN()];
                for (int j = 0; j < ml.getN(); j++) {
                    word[j] = ml.get(i, j).doubleValue();
                }
                codebook.add(word);
            }
            codebooks.put(n, codebook);
        }
        return codebooks;
    }

    public static void main( String[] args ) throws IOException, URISyntaxException, ClassNotFoundException {
        ViperParseApp app = new ViperParseApp();

        Map<Integer, List<double[]>> A = new HashMap<>();
        Map<Integer, List<double[]>> B = new HashMap<>();
        for (Map.Entry<Integer, List<double[]>> entry : app.codebook.entrySet()) {
            // replace codebook
            app.bowManager.getBow().getCodebooks().put(Feature.Type.ALL, entry.getValue());

            List<BowImage> bowImagesA = app.generateHist(testingA, maskA, "mask_");
            List<BowImage> bowImagesB = app.generateHist(testingB, maskB, "mask_");

            List<double[]> histA = bowImagesA.stream()
                    .map(bowImage -> bowImage.hist.get(Feature.Type.ALL))
                    .collect(Collectors.toList());
            List<double[]> histB = bowImagesB.stream()
                    .map(bowImage -> bowImage.hist.get(Feature.Type.ALL))
                    .collect(Collectors.toList());

            A.put(entry.getKey(), histA);
            B.put(entry.getKey(), histB);
        }

        List<double[]> histA = IntStream.range(0, A.get(0).size()).mapToObj(i -> {
            double[] doubles = Doubles.concat(A.get(0).get(i), A.get(1).get(i), A.get(2).get(i), A.get(3).get(i), A.get(4).get(i));

            // normalize
            double sum = Math.sqrt(new SumOfSquares().evaluate(doubles));
            for (int j = 0; j < doubles.length; j++) {
                doubles[j] = doubles[j] / sum;
            }
            return doubles;
        }).collect(Collectors.toList());

        List<double[]> histB = IntStream.range(0, B.get(0).size()).mapToObj(i -> {
            double[] doubles = Doubles.concat(B.get(0).get(i), B.get(1).get(i), B.get(2).get(i), B.get(3).get(i), B.get(4).get(i));

            // normalize
            double sum = Math.sqrt(new SumOfSquares().evaluate(doubles));
            for (int j = 0; j < doubles.length; j++) {
                doubles[j] = doubles[j] / sum;
            }
            return doubles;
        }).collect(Collectors.toList());

        // descriptor level non-pca
        double[] MR = app.viper.eval(histA, histB, false).getFirst();
        System.out.println("descriptorLevel not_pca:" + Doubles.asList(MR).subList(0, 50));

        // descriptor level pca
        MR = app.viper.eval(histA, histB, true).getFirst();
        System.out.println("descriptorLevel pca:" + Doubles.asList(MR).subList(0, 50));
    }
}
