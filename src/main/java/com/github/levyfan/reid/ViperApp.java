package com.github.levyfan.reid;

import com.github.levyfan.reid.eval.Viper;
import com.github.levyfan.reid.feature.Feature;
import com.google.common.primitives.Doubles;
import com.jmatio.io.MatFileWriter;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;

/**
 * @author fanliwen
 */
public class ViperApp extends App {

    private static final File testingA = new File("/data/reid/viper/cam_a/image");
    private static final File testingB = new File("/data/reid/viper/cam_b/image");
    private static final File maskA = new File("/data/reid/viper/mask/cam_a");
    private static final File maskB = new File("/data/reid/viper/mask/cam_b");

    private Viper viper = new Viper();

    private ViperApp(File codebookFile) throws IOException, URISyntaxException, ClassNotFoundException {
        super(codebookFile);
    }

    public static void main( String[] args ) throws IOException, URISyntaxException, ClassNotFoundException {
        ViperApp app = new ViperApp(new File(args[0]));

        List<BowImage> bowImagesA = app.generateHist(testingA, maskA, "mask_");
        List<BowImage> bowImagesB = app.generateHist(testingB, maskB, "mask_");

        // fusion
        List<double[]> histA = (List<double[]>) app.fusionHists(bowImagesA, types);
        List<double[]> histB = (List<double[]>) app.fusionHists(bowImagesB, types);

        // write to mat
        new MatFileWriter().write(
                "hist_" + numSuperpixels + "_" + compactness + ".mat",
                Arrays.asList(
                        com.github.levyfan.reid.util.MatrixUtils.to("HistA", histA),
                        com.github.levyfan.reid.util.MatrixUtils.to("HistB", histB)
                )
        );

        // descriptor level non-pca
        double[] MR = app.viper.eval(histA, histB, false).getFirst();
        System.out.println("descriptorLevel not_pca:" + Doubles.asList(MR).subList(0, 50));

        // descriptor level pca
        MR = app.viper.eval(histA, histB, true).getFirst();
        System.out.println("descriptorLevel pca:" + Doubles.asList(MR).subList(0, 50));

        // word level fusion
        if (wordLevel) {
            histA = (List<double[]>) app.fusionHists(bowImagesA, new Feature.Type[]{Feature.Type.ALL});
            histB = (List<double[]>) app.fusionHists(bowImagesB, new Feature.Type[]{Feature.Type.ALL});
            MR = app.viper.eval(histA, histB, false).getFirst();
            System.out.println("wordLevel:" + Doubles.asList(MR).subList(0, 50));
        }

        // separate
        RealMatrix[] scores = new RealMatrix[types.length];
        for (Feature.Type type : types) {
            histA = (List<double[]>) app.fusionHists(bowImagesA, new Feature.Type[]{type});
            histB = (List<double[]>) app.fusionHists(bowImagesB, new Feature.Type[]{type});
            Pair<double[], RealMatrix> pair = app.viper.eval(histA, histB, false);
            scores[type.ordinal()] = pair.getSecond();
            System.out.println(type + ":" + Doubles.asList(pair.getFirst()).subList(0,50));
        }

        // score level fusion
        RealMatrix score = MatrixUtils.createRealMatrix(
                scores[0].getRowDimension(), scores[0].getColumnDimension());
        for (int row = 0; row < score.getRowDimension(); row++) {
            for (int col = 0; col < score.getColumnDimension(); col++) {
                double value = scores[0].getEntry(row, col) * scores[1].getEntry(row, col)
                        * scores[2].getEntry(row, col) * scores[3].getEntry(row, col);
                score.setEntry(row, col, Math.sqrt(Math.sqrt(value)));
            }
        }
        System.out.println("scoreLevel:" + Doubles.asList(app.viper.eval(score)).subList(0,50));
    }
}
