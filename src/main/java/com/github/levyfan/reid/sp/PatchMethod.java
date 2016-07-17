package com.github.levyfan.reid.sp;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class PatchMethod implements SuperPixelMethond {

    private int size;

    public PatchMethod(int patchSize) {
        this.size = patchSize;
    }

    public SuperPixel[] patch(BufferedImage image) {
        List<SuperPixel> superPixels = new ArrayList<>();

        int label = 0;
        for (int i = 0; i < image.getHeight()/size; i++) {
            for (int j = 0; j < image.getWidth()/size; j++) {
                SuperPixel sp = new SuperPixel();
                sp.label = label;

                sp.rows = new int[size];
                sp.cols = new int[size];
                for (int k = 0; k < size; k++) {
                    sp.rows[k] = i*size + k;
                    sp.cols[k] = j*size + k;
                }

                label++;
                superPixels.add(sp);
            }
        }
        return superPixels.toArray(new SuperPixel[0]);
    }

    @Override
    public SuperPixel[] generate(BufferedImage image) {
        return patch(image);
    }
}
