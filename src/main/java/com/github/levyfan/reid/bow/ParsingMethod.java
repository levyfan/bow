package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.primitives.Ints;

import java.awt.image.BufferedImage;
import java.util.HashSet;
import java.util.Set;

/**
 * @author fanliwen
 */
public class ParsingMethod extends StripMethod {

    public ParsingMethod(int ystep, int length, int pstep) {
        super(ystep, length, pstep);
    }

    // 10, 20, 30, 51, 40, 61, 63

    public Strip[] strip(SuperPixel[] sp, BufferedImage mask) {
        ListMultimap<Integer, Integer> labels = ArrayListMultimap.create();

        for (int n = 0; n < sp.length; n++) {
            int[] rows = sp[n].rows;
            int[] cols = sp[n].cols;

            Set<Integer> labelsOfSuperpixel = new HashSet<>();
            for (int k = 0; k < rows.length; k++) {
                int value = mask.getRGB(cols[k], rows[k]) & 0x000000ff;
                if (value != 0) {
                    labelsOfSuperpixel.add(value);
                }
            }

            for (Integer integer : labelsOfSuperpixel) {
                labels.put(integer, n);
            }
        }

        Strip[] strips = new Strip[5];
        strips[0] = new Strip(0, Ints.toArray(labels.get(56)));
        strips[1] = new Strip(1, Ints.toArray(labels.get(79)));
        strips[2] = new Strip(2, Ints.toArray(labels.get(96)));
        strips[3] = new Strip(3, Ints.toArray(labels.get(110)));
        strips[4] = new Strip(4, Ints.toArray(labels.get(136)));
        return strips;
    }

    @Override
    public String toString() {
        return "ParsingMethod{}";
    }
}
