package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.util.MatrixUtils;
import com.google.common.io.PatternFilenameFilter;
import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author fanliwen
 */
public class MarketXqdaApp {

    private static final File training = new File("/data/reid/market1501/dataset/bounding_box_train");

    private static final PatternFilenameFilter filter =
            new PatternFilenameFilter("[^\\s]+(\\.(?i)(jpg|png|gif|bmp))$");

    public static void main(String[] args) throws IOException {
        MatFileReader reader = new MatFileReader(args[0]);
        MLArray mlArray = reader.getMLArray("Hist_train");
        RealMatrix hist = MatrixUtils.from((MLNumericArray) mlArray);

        // release memory
        mlArray.dispose();

        System.out.println("m=" + hist.getRowDimension());
        System.out.println("n=" + hist.getColumnDimension());

        File[] files = training.listFiles(filter);
        Arrays.sort(files);
        System.out.println(Arrays.toString(files));

        List<BowImage> bowImages = new ArrayList<>(files.length);
        for (int i = 0; i < files.length ; i ++) {
            File file = files[i];
            String[] strings = file.getName().split("_");
            String id = strings[0];
            String cam = strings[1].substring(1, 2);

            BowImage bowImage = new BowImage();
            bowImage.hist.put(Feature.Type.ALL, hist.getColumn(i));
            bowImage.id = String.valueOf(id);
            bowImage.cam = String.valueOf(cam);
            bowImages.add(bowImage);
        }

        // release memory
        hist = null;
        System.gc();

        Xqda xqda = new Xqda();
        RealMatrix M = xqda.apply(bowImages);

        System.out.println("M(0,1)=" + M.getEntry(0, 1));
        System.out.println("M(1,0)=" + M.getEntry(1, 0));
        new MatFileWriter().write("market1501_xqda.mat", Collections.singleton(new MLDouble("M", M.getData())));
    }
}
