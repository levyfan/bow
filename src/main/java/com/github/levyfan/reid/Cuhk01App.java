package com.github.levyfan.reid;

import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.collect.Lists;
import com.jmatio.io.MatFileWriter;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Cuhk01App extends App {

    private static final File testing = new File("/data/reid/cuhk01/campus");
    private static final File mask = new File("/data/reid/cuhk01/mask");

    private Cuhk01App(File codebookFile) throws IOException, URISyntaxException, ClassNotFoundException {
        super(codebookFile);
    }

    @Override
    List<BowImage> generateHist(File camFolder, File maskFoler, String maskPrefix) throws IOException, ClassNotFoundException {
        File[] camFiles = camFolder.listFiles(filter);
        Arrays.sort(camFiles);
        System.out.println(Arrays.toString(camFiles));

        return Lists.newArrayList(camFiles)
                .parallelStream()
                .map(camFile -> {
                    try {
                        System.out.println(camFile.getName());
                        BufferedImage image = ImageIO.read(camFile);

                        String maskFileName = maskPrefix + camFile.getName().replace("png", "bmp");
                        BufferedImage mask = ImageIO.read(new File(maskFoler, maskFileName));

                        BowImage bowImage = new BowImage(spMethod, stripMethod, image, mask);
                        bowImage.id = camFile.getName().substring(0, 4);
                        bowImage.cam = String.valueOf(
                                (Integer.valueOf(camFile.getName().substring(4, 7)) + 1) / 2);

                        bowManager.bow(bowImage);
                        return bowImage;
                    } catch (Exception e) {
                        System.err.println("cam file = " + camFile);
                        throw new RuntimeException(e);
                    }
                }).collect(Collectors.toList());
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, URISyntaxException {
        Cuhk01App app = new Cuhk01App(new File(args[0]));

        List<BowImage> bowImages = app.generateHist(testing, mask, "");

        List<double[]> hist = (List<double[]>) app.fusionHists(bowImages, types);

        // write to mat
        new MatFileWriter().write(
                "cuhk01_hist_" + numSuperpixels + "_" + compactness + ".mat",
                Collections.singletonList(MatrixUtils.to("hist", hist))
        );
    }
}
