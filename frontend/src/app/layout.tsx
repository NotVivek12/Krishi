import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AgriFL - Crop Recommendation",
  description: "Privacy-Preserving Intelligent Crop Recommendation using Federated Learning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-50 text-slate-900 flex flex-col min-h-screen`}>
        <nav className="bg-emerald-700 text-white shadow-md">
          <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
            <Link href="/" className="text-xl font-bold tracking-tight">
              AgriFL <span className="font-light text-emerald-200 text-sm hidden sm:inline ml-2">Federated Crop Recommender</span>
            </Link>
            <div className="space-x-6 text-sm font-medium">
              <Link href="/" className="hover:text-emerald-200 transition-colors">Dashboard</Link>
              <Link href="/predict" className="hover:text-emerald-200 transition-colors">Predict</Link>
              <Link href="/history" className="hover:text-emerald-200 transition-colors">History</Link>
              <Link href="/about" className="hover:text-emerald-200 transition-colors">About</Link>
            </div>
          </div>
        </nav>
        <main className="flex-grow max-w-6xl w-full mx-auto px-4 py-8">
          {children}
        </main>
        <footer className="bg-white border-t py-6 text-center text-sm text-slate-500">
          <p>AgriFL Final Year Project &copy; {new Date().getFullYear()}</p>
        </footer>
      </body>
    </html>
  );
}
