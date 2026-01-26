import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { LocaleProvider } from './contexts/LocaleContext';

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Total Segmentator Web App",
  description: "Automated medical imaging segmentation with Blender 3D visualization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <LocaleProvider>
          {children}
        </LocaleProvider>
      </body>
    </html>
  );
}
