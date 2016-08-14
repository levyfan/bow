package com.github.levyfan.reid.bow;

/**
 * @author fanliwen
 */
public class Strip {
    public int index;
    public int[] superPixels;

    public int[] lowSuperPixels;
    public int[] highSuperPixels;

    public Strip(int index, int[] superPixels) {
        this.index = index;
        this.superPixels = superPixels;

        this.lowSuperPixels = superPixels;
        this.highSuperPixels = superPixels;
    }
}
