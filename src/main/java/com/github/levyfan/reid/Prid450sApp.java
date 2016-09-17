package com.github.levyfan.reid;

import com.google.common.collect.Lists;
import com.google.common.io.PatternFilenameFilter;
import com.jmatio.io.MatFileWriter;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Prid450sApp extends App {

    private static final File testingA = new File("/data/reid/prid_450s_resize/cam_a");
    private static final File testingB = new File("/data/reid/prid_450s_resize/cam_b");

    private Prid450sApp(File codebookFile) throws IOException, URISyntaxException, ClassNotFoundException {
        super(codebookFile);
    }

    @Override
    List<BowImage> generateHist(File camFolder, File maskFoler, String maskPrefix) throws IOException, ClassNotFoundException {
        File[] camFiles = camFolder.listFiles(new PatternFilenameFilter("img_.*\\.bmp"));
        Arrays.sort(camFiles);
        System.out.println(Arrays.toString(camFiles));

        return Lists.newArrayList(camFiles)
                .parallelStream()
                .map(camFile -> {
                    try {
                        System.out.println(camFile.getName());
                        BufferedImage image = ImageIO.read(camFile);

                        String maskFileName = maskPrefix + camFile.getName().replace("img", "man");
                        BufferedImage mask = ImageIO.read(new File(maskFoler, maskFileName));

                        BowImage bowImage = new BowImage(spMethod, stripMethod, image, mask);
                        bowImage.id = camFile.getName().split("\\.")[0].split("_")[1];

                        bowManager.bow(bowImage);
                        return bowImage;
                    } catch (Exception e) {
                        System.err.println("cam file = " + camFile);
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main( String[] args ) throws IOException, URISyntaxException, ClassNotFoundException {
        Prid450sApp app = new Prid450sApp(new File(args[0]));

        List<BowImage> bowImagesA = app.generateHist(testingA, testingA, "");
        List<BowImage> bowImagesB = app.generateHist(testingB, testingB, "");

        List<double[]> histA = (List<double[]>) app.fusion(bowImagesA, types);
        List<double[]> histB = (List<double[]>) app.fusion(bowImagesB, types);

        // write to mat
        new MatFileWriter().write(
                "prid450s_hist_" + numSuperpixels + "_" + compactness + ".mat",
                Arrays.asList(
                        com.github.levyfan.reid.util.MatrixUtils.to("HistA", histA),
                        com.github.levyfan.reid.util.MatrixUtils.to("HistB", histB)
                )
        );
    }
}
