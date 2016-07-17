package com.github.levyfan.reid.sp;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.primitives.Ints;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

/**
 * @author fanliwen
 */
public class Slic implements SuperPixelMethond {

    private static final double Xr = 0.950456;    //reference white
    private static final double Yr = 1.0;        //reference white
    private static final double Zr = 1.088754;    //reference white

    private static final double epsilon = 0.008856;    //actual CIE standard
    private static final double kappa = 903.3;        //actual CIE standard

    private int numSuperpixels;
    private double compactness;

    public Slic(int numSuperpixels, double compactness) {
        this.numSuperpixels = numSuperpixels;
        this.compactness = compactness;
    }

    public SuperPixel[] slic(BufferedImage image) {
        int sz = image.getHeight() * image.getWidth();

        //---------------------------
        // Perform color conversion
        //---------------------------
        double[] lvec = new double[sz];
        double[] avec = new double[sz];
        double[] bvec = new double[sz];
        rgbtolab(image, lvec, avec, bvec);

        //---------------------------
        // Find seeds
        //---------------------------
        int[] seedIndices = new int[sz];
        int numseeds = getLABXYSeeds(numSuperpixels, image.getWidth(), image.getHeight(), seedIndices);
        double[] kseedsx = new double[numseeds];
        double[] kseedsy = new double[numseeds];
        double[] kseedsl = new double[numseeds];
        double[] kseedsa = new double[numseeds];
        double[] kseedsb = new double[numseeds];
        for (int k = 0; k < numseeds; k++) {
            kseedsx[k] = seedIndices[k] % image.getWidth();
            kseedsy[k] = seedIndices[k] / image.getWidth();
            kseedsl[k] = lvec[seedIndices[k]];
            kseedsa[k] = avec[seedIndices[k]];
            kseedsb[k] = bvec[seedIndices[k]];
        }

        //---------------------------
        // Compute superpixels
        //---------------------------
        int[] klabels = new int[sz];
        performSuperpixelSLIC(
                lvec, avec, bvec,
                kseedsl, kseedsa, kseedsb, kseedsx, kseedsy,
                image.getWidth(), image.getHeight(), numseeds,
                klabels, numSuperpixels, compactness);

        //---------------------------
        // Enforce connectivity
        //---------------------------
        int[] clabels = new int[sz];
        int finalNumberOfLabels = enforceSuperpixelConnectivity(
                klabels, image.getWidth(), image.getHeight(), numSuperpixels, clabels);

        RealMatrix outlabels = MatrixUtils.createRealMatrix(image.getHeight(), image.getWidth());
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                outlabels.setEntry(y, x, clabels[y * image.getWidth() + x]);
            }
        }

        ListMultimap<Integer, Integer> rows = ArrayListMultimap.create();
        ListMultimap<Integer, Integer> columns = ArrayListMultimap.create();
        for (int i = 0; i < outlabels.getRowDimension(); i++) {
            for (int j = 0; j < outlabels.getColumnDimension(); j++) {
                int label = (int) outlabels.getEntry(i, j);
                rows.put(label, i);
                columns.put(label, j);
            }
        }

        SuperPixel[] superPixels = new SuperPixel[finalNumberOfLabels];
        for (int label = 0; label < finalNumberOfLabels; label++) {
            SuperPixel superPixel = new SuperPixel();
            superPixel.label = label;
            superPixel.rows = Ints.toArray(rows.get(label));
            superPixel.cols = Ints.toArray(columns.get(label));
            superPixels[label] = superPixel;
        }
        return superPixels;
    }

    private void rgbtolab(BufferedImage image, double[] lvec, double[] avec, double[] bvec) {
        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                Color color = new Color(image.getRGB(x, y));
                double R = color.getRed() / 255.0;
                double G = color.getGreen() / 255.0;
                double B = color.getBlue() / 255.0;

                double r = (R <= 0.04045) ? (R / 12.92) : Math.pow((R + 0.055) / 1.055, 2.4);
                double g = (G <= 0.04045) ? (G / 12.92) : Math.pow((G + 0.055) / 1.055, 2.4);
                double b = (B <= 0.04045) ? (B / 12.92) : Math.pow((B + 0.055) / 1.055, 2.4);

                //------------------------
                // XYZ to LAB conversion
                //------------------------
                double xr = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / Xr;
                double yr = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / Yr;
                double zr = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / Zr;

                double fx = (xr > epsilon) ? Math.pow(xr, 1.0 / 3.0) : (kappa * xr + 16.0) / 116.0;
                double fy = (yr > epsilon) ? Math.pow(yr, 1.0 / 3.0) : (kappa * yr + 16.0) / 116.0;
                double fz = (zr > epsilon) ? Math.pow(zr, 1.0 / 3.0) : (kappa * zr + 16.0) / 116.0;

                double lval = 116.0 * fy - 16.0;
                double aval = 500.0 * (fx - fy);
                double bval = 200.0 * (fy - fz);

                lvec[y * image.getWidth() + x] = lval;
                avec[y * image.getWidth() + x] = aval;
                bvec[y * image.getWidth() + x] = bval;
            }
        }
    }

    private int getLABXYSeeds(int numSuperpixels, int width, int height, int[] seedIndices) {
        int step = (int) (Math.sqrt((double) (width * height) / (double) (numSuperpixels)) + 0.5);

        int xstrips = (int) (0.5 + (double) (width) / (double) (step));
        int ystrips = (int) (0.5 + (double) (height) / (double) (step));

        int xerr = width - step * xstrips;
        if (xerr < 0) {
            xstrips--;
            xerr = width - step * xstrips;
        }

        int yerr = height - step * ystrips;
        if (yerr < 0) {
            ystrips--;
            yerr = height - step * ystrips;
        }

        double xerrperstrip = (double) (xerr) / (double) (xstrips);
        double yerrperstrip = (double) (yerr) / (double) (ystrips);

        int xoff = step / 2;
        int yoff = step / 2;

        int n = 0;
        for (int y = 0; y < ystrips; y++) {
            int ye = (int) (y * yerrperstrip);
            for (int x = 0; x < xstrips; x++) {
                int xe = (int) (x * xerrperstrip);
                int seedx = (x * step + xoff + xe);
                int seedy = (y * step + yoff + ye);
                seedIndices[n] = seedy * width + seedx;
                n++;
            }
        }
        return n;
    }

    private void performSuperpixelSLIC(
            double[] lvec, double[] avec, double[] bvec,
            double[] kseedsl, double[] kseedsa, double[] kseedsb, double[] kseedsx, double[] kseedsy,
            int width, int height, int numseeds,
            int[] klabels, int numSuperpixels, double compactness) {
        int sz = width * height;
        int offset = (int) (Math.sqrt((double) (width * height) / (double) (numSuperpixels)) + 0.5);

        double[] clustersize = new double[numseeds];
        double[] inv = new double[numseeds];
        double[] sigmal = new double[numseeds];
        double[] sigmaa = new double[numseeds];
        double[] sigmab = new double[numseeds];
        double[] sigmax = new double[numseeds];
        double[] sigmay = new double[numseeds];
        double[] distvec = new double[sz];
        double invwt = 1.0 / ((offset / compactness) * (offset / compactness));

        for (int itr = 0; itr < 10; itr++) {
            Arrays.fill(distvec, Double.MAX_VALUE);

            for (int n = 0; n < numseeds; n++) {
                int x1 = Math.max((int) (kseedsx[n] - offset), 0);
                int y1 = Math.max((int) (kseedsy[n] - offset), 0);
                int x2 = Math.min((int) (kseedsx[n] + offset), width);
                int y2 = Math.min((int) (kseedsy[n] + offset), height);

                for (int y = y1; y < y2; y++) {
                    for (int x = x1; x < x2; x++) {
                        int i = y * width + x;

                        double l = lvec[i];
                        double a = avec[i];
                        double b = bvec[i];

                        double dist = (l - kseedsl[n]) * (l - kseedsl[n]) +
                                (a - kseedsa[n]) * (a - kseedsa[n]) +
                                (b - kseedsb[n]) * (b - kseedsb[n]);

                        double distxy = (x - kseedsx[n]) * (x - kseedsx[n]) + (y - kseedsy[n]) * (y - kseedsy[n]);

                        dist += distxy * invwt;

                        if (dist < distvec[i]) {
                            distvec[i] = dist;
                            klabels[i] = n;
                        }
                    }
                }
            }
            //-----------------------------------------------------------------
            // Recalculate the centroid and store in the seed values
            //-----------------------------------------------------------------
            Arrays.fill(sigmal, 0);
            Arrays.fill(sigmaa, 0);
            Arrays.fill(sigmab, 0);
            Arrays.fill(sigmax, 0);
            Arrays.fill(sigmay, 0);
            Arrays.fill(clustersize, 0);

            int ind = 0;
            for (int r = 0; r < height; r++) {
                for (int c = 0; c < width; c++) {
                    if (klabels[ind] >= 0) {
                        sigmal[klabels[ind]] += lvec[ind];
                        sigmaa[klabels[ind]] += avec[ind];
                        sigmab[klabels[ind]] += bvec[ind];
                        sigmax[klabels[ind]] += c;
                        sigmay[klabels[ind]] += r;
                        clustersize[klabels[ind]] += 1.0;
                    }
                    ind++;
                }
            }

            for (int k = 0; k < numseeds; k++) {
                if (clustersize[k] <= 0) clustersize[k] = 1;
                inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
            }

            for (int k = 0; k < numseeds; k++) {
                kseedsl[k] = sigmal[k] * inv[k];
                kseedsa[k] = sigmaa[k] * inv[k];
                kseedsb[k] = sigmab[k] * inv[k];
                kseedsx[k] = sigmax[k] * inv[k];
                kseedsy[k] = sigmay[k] * inv[k];
            }
        }
    }

    private int enforceSuperpixelConnectivity(
            int[] labels, int width, int height, int numSuperpixels, int[] nlabels) {
        int[] dx4 = {-1, 0, 1, 0};
        int[] dy4 = {0, -1, 0, 1};
        int sz = width * height;
        int SUPSZ = sz / numSuperpixels;
        int[] xvec = new int[SUPSZ * 10];
        int[] yvec = new int[SUPSZ * 10];

        Arrays.fill(nlabels, -1);
        int oindex = 0;
        int adjlabel = 0;//adjacent label
        int label = 0;
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if (0 > nlabels[oindex]) {
                    nlabels[oindex] = label;
                    //--------------------
                    // Start a new segment
                    //--------------------
                    xvec[0] = k;
                    yvec[0] = j;
                    //-------------------------------------------------------
                    // Quickly find an adjacent label for use later if needed
                    //-------------------------------------------------------
                    for (int n = 0; n < 4; n++) {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
                            int nindex = y * width + x;
                            if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                        }
                    }

                    int count = 1;
                    for (int c = 0; c < count; c++) {
                        for (int n = 0; n < 4; n++) {
                            int x = xvec[c] + dx4[n];
                            int y = yvec[c] + dy4[n];

                            if ((x >= 0 && x < width) && (y >= 0 && y < height)) {
                                int nindex = y * width + x;

                                if (0 > nlabels[nindex] && labels[oindex] == labels[nindex]) {
                                    xvec[count] = x;
                                    yvec[count] = y;
                                    nlabels[nindex] = label;
                                    count++;
                                }
                            }

                        }
                    }
                    //-------------------------------------------------------
                    // If segment size is less then a limit, assign an
                    // adjacent label found before, and decrement label count.
                    //-------------------------------------------------------
                    if (count <= SUPSZ >> 2) {
                        for (int c = 0; c < count; c++) {
                            int ind = yvec[c] * width + xvec[c];
                            nlabels[ind] = adjlabel;
                        }
                        label--;
                    }
                    label++;
                }
                oindex++;
            }
        }
        return label;
    }

    @Override
    public SuperPixel[] generate(BufferedImage image) {
        return slic(image);
    }
}
