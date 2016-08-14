package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import org.apache.commons.math3.stat.descriptive.moment.Mean;

import java.awt.image.BufferedImage;
import java.util.HashSet;
import java.util.Set;

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

            Set<Integer> labelsOfSuperpixel = new HashSet<>();
            for (int k = 0; k < rows.length; k++) {
                int value = mask.getRGB(cols[k], rows[k]) & 0x000000ff;
                switch (value) {
                    case 0: break;
                    case 56: labelsOfSuperpixel.add(0); break;
                    case 79: labelsOfSuperpixel.add(0); break;
                    case 96: labelsOfSuperpixel.add(0); break;
                    case 110: labelsOfSuperpixel.add(0); break;
                    case 136: labelsOfSuperpixel.add(0); break;
                    default: labelsOfSuperpixel.add(0); break;
                }
            }

            double cRow = new Mean().evaluate(Doubles.toArray(Ints.asList(rows)));

            // nstrip*ystep <= row < nstrip*ystep + length
            // (row-length)/ystep < nstrip <= row/ystep
            int start = (int) Math.max(Math.floor((cRow-length+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor(cRow/ystep); nstrip++) {
                for (int labelOfSuperpixel : labelsOfSuperpixel) {
                    labels.put(nstrip + labelOfSuperpixel, n);
                }
            }

            // pooling low
            // nstrip*ystep - pstep <= row < nstrip*ystep + length - pstep
            // (row - length + pstep)/ystep < nstrip <= (row + pstep)/ystep
            start = (int) Math.max(Math.floor((cRow-length+pstep+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor((cRow+pstep)/ystep); nstrip++) {
                for (int labelOfSuperpixel : labelsOfSuperpixel) {
                    lowLabels.put(nstrip + labelOfSuperpixel, n);
                }
            }

            // pooling high
            // nstrip*ystep + pstep <= row < nstrip*ystep + length + pstep
            // (row - length - pstep)/ystep < nstrip <= (row - pstep)/ystep
            start = (int) Math.max(Math.floor((cRow-length-pstep+ystep)/ystep), 0);
            for (int nstrip = start; nstrip <= (int) Math.floor((cRow-pstep)/ystep); nstrip++) {
                for (int labelOfSuperpixel : labelsOfSuperpixel) {
                    highLabels.put(nstrip + labelOfSuperpixel, n);
                }
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
