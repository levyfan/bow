package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.awt.image.BufferedImage;

/**
 * @author fanliwen
 */
public class StripMethod {

    private int ystep;

    private int length;

    private int pstep;

    public StripMethod(int ystep, int length, int pstep) {
        this.ystep = ystep;
        this.length = length;
        this.pstep = pstep;
    }

    public Strip[] strip(SuperPixel[] sp, BufferedImage mask) {
        ListMultimap<Integer, Integer> labels = ArrayListMultimap.create();
        ListMultimap<Integer, Integer> lowLabels = ArrayListMultimap.create();
        ListMultimap<Integer, Integer> highLabels = ArrayListMultimap.create();
        for (int n = 0; n < sp.length; n++) {
            int[] rows = sp[n].rows;
            int[] cols = sp[n].cols;

            int sum = 0;
            for (int k = 0; k < rows.length; k++) {
                sum += (mask.getRGB(cols[k], rows[k]) & 0x00ffffff);
            }
            if (sum == 0) {
                // background
                continue;
            }

            double cRow = new Mean().evaluate(Doubles.toArray(Ints.asList(rows)));

            // nstrip*ystep <= row < nstrip*ystep + length
            // (row-length)/ystep < nstrip <= row/ystep
            int start = (int) Math.max(Math.floor((cRow-length+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor(cRow/ystep); nstrip++) {
                labels.put(nstrip, n);
            }

            // pooling low
            // nstrip*ystep - pstep <= row < nstrip*ystep + length - pstep
            // (row - length + pstep)/ystep < nstrip <= (row + pstep)/ystep
            start = (int) Math.max(Math.floor((cRow-length+pstep+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor((cRow+pstep)/ystep); nstrip++) {
                lowLabels.put(nstrip, n);
            }

            // pooling high
            // nstrip*ystep + pstep <= row < nstrip*ystep + length + pstep
            // (row - length - pstep)/ystep < nstrip <= (row - pstep)/ystep
            start = (int) Math.max(Math.floor((cRow-length-pstep+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor((cRow-pstep)/ystep); nstrip++) {
                highLabels.put(nstrip, n);
            }
        }

        int stripNumber = (mask.getHeight()-length)/ystep + 1;
        Strip[] strips = new Strip[stripNumber];
        for (int nstrip = 0; nstrip < stripNumber; nstrip++) {
            Strip strip = new Strip(nstrip, Ints.toArray(labels.get(nstrip)));

            // pooling
            strip.lowSuperPixels = Ints.toArray(lowLabels.get(nstrip));
            strip.highSuperPixels = Ints.toArray(highLabels.get(nstrip));

            strips[nstrip] = strip;
        }
        return strips;
    }

    @Override
    public String toString() {
        return "StripMethod{" +
                "ystep=" + ystep +
                ", length=" + length +
                ", pstep=" + pstep +
                '}';
    }
}
